from pydantic import BaseModel


class ProjectCreate(BaseModel):

    title: str

    description: str | None = None

    objective: str | None = None


class ProjectResponse(ProjectCreate):

    id: int

    status: str

    created_at: str

    class Config:
        from_attributes = True