import wikipediaapi
from pydantic import BaseModel
import re

class InstitutionDetails(BaseModel):
    name: str
    founder: str
    founded_year: str
    branches: str
    employees: str
    summary: str  # fixed indentation

def fetch_wikipedia_summary(institution_name):
    """
    Fetches the Wikipedia summary of an institution.
    """
    wiki_wiki = wikipediaapi.Wikipedia("en")
    page = wiki_wiki.page(institution_name)
    if not page.exists():  # fixed indentation
        raise ValueError(f"Wikipedia page for '{institution_name}' not found.")
    return page.summary

def extract_institution_details(institution_name):
    """
    Extracts institution-related details from the Wikipedia summary.
    """
    summary = fetch_wikipedia_summary(institution_name)
    founder = re.search(r"Founded by (.*?)[.,]", summary)
    founded_year = re.search(r"Founded in (\d{4})", summary)
    branches = re.search(r"has (\d+) branches", summary)
    employees = re.search(r"employs approximately (\d+)", summary)

    details = InstitutionDetails(
        name=institution_name,
        founder=founder.group(1) if founder else "Not Found",
        founded_year=founded_year.group(1) if founded_year else "Not Found",
        branches=branches.group(1) if branches else "Not Found",
        employees=employees.group(1) if employees else "Not Found",
        summary=" ".join(summary.split(".")[:4])  # First 4 sentences
    )
    return details  # fixed indentation

if __name__ == "__main__":
    institution_name = input("Enter the Institution Name: ")
    try:
        details = extract_institution_details(institution_name)
        print("\nExtracted Institution Details:")
        print(details.json(indent=4))
    except Exception as e:
        print(f"Error: {e}")