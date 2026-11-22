"""Strict public contract for credit-applicant data."""
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictFloat, field_validator
import math


class Applicant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: StrictInt = Field(ge=18, le=100)
    revenu_mensuel: StrictFloat | StrictInt = Field(gt=0, le=100_000_000)
    montant_credit: StrictFloat | StrictInt = Field(gt=0, le=1_000_000_000)
    duree_mois: StrictInt = Field(ge=1, le=360)
    anciennete_activite: StrictFloat | StrictInt = Field(ge=0, le=82)

    @field_validator('anciennete_activite')
    @classmethod
    def experience_not_more_than_working_life(cls, value, info):
        age = info.data.get('age')
        if age is not None and value > age - 16:
            raise ValueError('anciennete_activite incoherente avec age')
        return value

    @field_validator('revenu_mensuel', 'montant_credit', 'anciennete_activite')
    @classmethod
    def finite(cls, value):
        if not math.isfinite(value):
            raise ValueError('valeur non finie interdite')
        return value
