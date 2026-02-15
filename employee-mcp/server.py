import os
import sqlite3
from contextlib import contextmanager
from typing import Optional

from fastmcp import FastMCP

from init_db import DB_PATH, init_db

# ── Bootstrap ──────────────────────────────────────────────────────────────
if not os.path.exists(DB_PATH):
    init_db()

mcp = FastMCP("Employee Directory")


# ── DB helper ──────────────────────────────────────────────────────────────
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)


# ── Tools ──────────────────────────────────────────────────────────────────

@mcp.tool()
def list_employees(active_only: bool = True) -> list[dict]:
    """Return all employees, optionally filtered to active ones only."""
    with get_db() as conn:
        query = """
            SELECT e.id, e.first_name, e.last_name, e.email, e.phone,
                   d.name AS department, e.job_title, e.salary,
                   e.hire_date, e.is_active
            FROM employees e
            LEFT JOIN departments d ON d.id = e.department_id
        """
        if active_only:
            query += " WHERE e.is_active = 1"
        query += " ORDER BY e.last_name, e.first_name"
        rows = conn.execute(query).fetchall()
        return [row_to_dict(r) for r in rows]


@mcp.tool()
def get_employee(employee_id: int) -> dict:
    """Return a single employee record by ID."""
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT e.id, e.first_name, e.last_name, e.email, e.phone,
                   d.name AS department, e.job_title, e.salary,
                   e.hire_date, e.is_active
            FROM employees e
            LEFT JOIN departments d ON d.id = e.department_id
            WHERE e.id = ?
            """,
            (employee_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"No employee found with id={employee_id}")
    return row_to_dict(row)


@mcp.tool()
def search_employees(query: str) -> list[dict]:
    """Search employees by first name, last name, email, or job title (case-insensitive)."""
    pattern = f"%{query}%"
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT e.id, e.first_name, e.last_name, e.email, e.phone,
                   d.name AS department, e.job_title, e.salary,
                   e.hire_date, e.is_active
            FROM employees e
            LEFT JOIN departments d ON d.id = e.department_id
            WHERE e.first_name  LIKE ?
               OR e.last_name   LIKE ?
               OR e.email       LIKE ?
               OR e.job_title   LIKE ?
            ORDER BY e.last_name, e.first_name
            """,
            (pattern, pattern, pattern, pattern),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


@mcp.tool()
def list_departments() -> list[dict]:
    """Return all departments with employee headcount."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT d.id, d.name,
                   COUNT(e.id)                                       AS total_employees,
                   SUM(CASE WHEN e.is_active = 1 THEN 1 ELSE 0 END) AS active_employees
            FROM departments d
            LEFT JOIN employees e ON e.department_id = d.id
            GROUP BY d.id, d.name
            ORDER BY d.name
            """
        ).fetchall()
    return [row_to_dict(r) for r in rows]


@mcp.tool()
def get_employees_by_department(department_name: str, active_only: bool = True) -> list[dict]:
    """Return all employees in a given department (case-insensitive name match)."""
    with get_db() as conn:
        query = """
            SELECT e.id, e.first_name, e.last_name, e.email, e.phone,
                   d.name AS department, e.job_title, e.salary,
                   e.hire_date, e.is_active
            FROM employees e
            JOIN departments d ON d.id = e.department_id
            WHERE LOWER(d.name) = LOWER(?)
        """
        params: list = [department_name]
        if active_only:
            query += " AND e.is_active = 1"
        query += " ORDER BY e.last_name, e.first_name"
        rows = conn.execute(query, params).fetchall()
    return [row_to_dict(r) for r in rows]


@mcp.tool()
def get_salary_stats(department_name: Optional[str] = None) -> dict:
    """Return min / max / average salary, optionally scoped to a department."""
    with get_db() as conn:
        if department_name:
            row = conn.execute(
                """
                SELECT MIN(e.salary) AS min_salary,
                       MAX(e.salary) AS max_salary,
                       ROUND(AVG(e.salary), 2) AS avg_salary,
                       COUNT(*) AS employee_count
                FROM employees e
                JOIN departments d ON d.id = e.department_id
                WHERE LOWER(d.name) = LOWER(?) AND e.is_active = 1
                """,
                (department_name,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT MIN(salary) AS min_salary,
                       MAX(salary) AS max_salary,
                       ROUND(AVG(salary), 2) AS avg_salary,
                       COUNT(*) AS employee_count
                FROM employees
                WHERE is_active = 1
                """
            ).fetchone()
    return row_to_dict(row)


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Runs as an HTTP server on http://localhost:8000/mcp
    mcp.run(transport="http", host="0.0.0.0", port=8000)
