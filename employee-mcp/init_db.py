import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "employees.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS departments (
            id   INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS employees (
            id            INTEGER PRIMARY KEY,
            first_name    TEXT NOT NULL,
            last_name     TEXT NOT NULL,
            email         TEXT NOT NULL UNIQUE,
            phone         TEXT,
            department_id INTEGER REFERENCES departments(id),
            job_title     TEXT NOT NULL,
            salary        REAL NOT NULL,
            hire_date     TEXT NOT NULL,
            is_active     INTEGER NOT NULL DEFAULT 1
        );
    """)

    departments = [
        (1, "Engineering"),
        (2, "Product"),
        (3, "Design"),
        (4, "Marketing"),
        (5, "HR"),
        (6, "Finance"),
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO departments (id, name) VALUES (?, ?)", departments
    )

    employees = [
        (1,  "Alice",   "Johnson",  "alice.johnson@example.com",  "555-0101", 1, "Senior Software Engineer", 135000, "2021-03-15", 1),
        (2,  "Bob",     "Smith",    "bob.smith@example.com",      "555-0102", 1, "Backend Engineer",         115000, "2022-07-01", 1),
        (3,  "Carol",   "Williams", "carol.williams@example.com", "555-0103", 1, "DevOps Engineer",          120000, "2020-11-20", 1),
        (4,  "David",   "Brown",    "david.brown@example.com",    "555-0104", 2, "Product Manager",          130000, "2019-05-10", 1),
        (5,  "Eva",     "Davis",    "eva.davis@example.com",      "555-0105", 2, "Associate PM",              95000, "2023-01-17", 1),
        (6,  "Frank",   "Miller",   "frank.miller@example.com",   "555-0106", 3, "Lead Designer",            110000, "2020-08-03", 1),
        (7,  "Grace",   "Wilson",   "grace.wilson@example.com",   "555-0107", 3, "UX Researcher",             98000, "2022-04-25", 1),
        (8,  "Henry",   "Moore",    "henry.moore@example.com",    "555-0108", 4, "Marketing Director",       145000, "2018-02-14", 1),
        (9,  "Iris",    "Taylor",   "iris.taylor@example.com",    "555-0109", 4, "Content Strategist",        88000, "2023-06-05", 1),
        (10, "Jack",    "Anderson", "jack.anderson@example.com",  "555-0110", 5, "HR Manager",               105000, "2019-09-22", 1),
        (11, "Karen",   "Thomas",   "karen.thomas@example.com",   "555-0111", 5, "Recruiter",                 78000, "2022-11-30", 1),
        (12, "Leo",     "Jackson",  "leo.jackson@example.com",    "555-0112", 6, "CFO",                      200000, "2017-01-09", 1),
        (13, "Mia",     "White",    "mia.white@example.com",      "555-0113", 6, "Financial Analyst",         92000, "2021-08-18", 1),
        (14, "Nathan",  "Harris",   "nathan.harris@example.com",  "555-0114", 1, "Junior Engineer",           85000, "2024-02-12", 1),
        (15, "Olivia",  "Martin",   "olivia.martin@example.com",  "555-0115", 2, "Senior PM",                140000, "2020-03-30", 0),
    ]
    cur.executemany(
        """INSERT OR IGNORE INTO employees
           (id, first_name, last_name, email, phone, department_id,
            job_title, salary, hire_date, is_active)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        employees,
    )

    conn.commit()
    conn.close()
    print(f"Database initialised at {DB_PATH}")


if __name__ == "__main__":
    init_db()
