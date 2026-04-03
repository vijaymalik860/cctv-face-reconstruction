"""
Database Setup
SQLAlchemy engine, session, and initialization for PostgreSQL.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import DATABASE_URL

# Create engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base for models
Base = declarative_base()


def get_db():
    """Dependency for FastAPI - yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables defined by SQLAlchemy models."""
    from app.models.schemas import ProcessingJob, FaceRegion  # noqa: F401
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")
