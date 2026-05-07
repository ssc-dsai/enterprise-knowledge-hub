"""Python file for migration runs through pipeline"""
import logging

from repository.database import db, initialize_database
from repository.migration.initial_baseline import run_init_migration

def main() -> None:
    """Main method"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.info("Running database migrations")
    initialize_database()
    run_init_migration(db)
    logging.info("Database migrations completed")


if __name__ == "__main__":
    main()
