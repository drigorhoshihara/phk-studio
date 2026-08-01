from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text

from database.database import Base


class Project(Base):

    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)

    title = Column(String(250), nullable=False)

    description = Column(Text)

    objective = Column(Text)

    status = Column(String(50), default="Novo")

    created_at = Column(String(50))