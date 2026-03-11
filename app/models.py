from sqlalchemy import Column, Integer, Text
from app.database import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    response = Column(Text, nullable=False)