import logging

from repository.database import migrate_database

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.info("Running database migrations")
    migrate_database()
    logging.info("Database migrations completed")


if __name__ == "__main__":
    main()
