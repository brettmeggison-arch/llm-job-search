import os
import requests
import PyPDF2
from bs4 import BeautifulSoup
from openai import OpenAI

# Initialize the LLM client
# Ensure your environment variable is set: export OPENAI_API_KEY="your_key"
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def extract_resume_from_pdf(pdf_path):
    """Reads and extracts text from the provided PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    except FileNotFoundError:
        print(f"Error: Could not find the PDF file at '{pdf_path}'.")
        exit(1)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        exit(1)

def evaluate_job(job, resume_text):
    """
    Passes the job description and dynamic resume text to the LLM to calculate a match score.
    """
    prompt = f"""
    You are an expert technical recruiter. Evaluate the following job description against my resume.
    Score the match from 1 to 10, where 10 is a perfect fit.
    Provide a concise, 2-sentence reasoning for your score based on overlapping skills or missing requirements.

    Resume:
    {resume_text}

    Job Title: {job['title']}
    Company: {job['company']}
    Job Description: {job['description']}

    Format your output exactly as follows:
    Score: [1-10]
    Reasoning: [Your analysis]
    """

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a helpful AI recruiting assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error evaluating job: {e}"


def scrape_jobs(keywords, url="https://realpython.github.io/fake-jobs/"):
    """
    Scrapes the job board and filters results based on user-specified keywords.
    """
    print(f"Fetching jobs from {url}...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to retrieve webpage. Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    job_elements = soup.find_all("div", class_="card-content")

    jobs = []
    # Convert keywords to lowercase for case-insensitive matching
    keywords_lower = [k.lower() for k in keywords]

    for element in job_elements:
        title_element = element.find("h2", class_="title")
        company_element = element.find("h3", class_="company")
        description_element = element.find("p", class_="location")

        title = title_element.text.strip() if title_element else "Unknown Title"
        company = company_element.text.strip() if company_element else "Unknown Company"
        description = description_element.text.strip() if description_element else "No description provided."

        # Combine the title and description to search for the keywords
        search_text = f"{title} {description}".lower()

        # Only add the job if at least one keyword is found in the text
        if any(keyword in search_text for keyword in keywords_lower):
            jobs.append({
                "title": title,
                "company": company,
                "description": description
            })

            # Stop after finding 5 matching jobs to save API credits
            if len(jobs) >= 5:
                break

    return jobs


def generate_keywords_from_resume(resume_text):
    """
    Asks the LLM to generate targeted job search keywords based on the resume.
    """
    prompt = f"""
    Analyze the following resume and identify the 5 most important keywords 
    to use for a job board search. Include a mix of ideal job titles and core technologies.
    Return ONLY a comma-separated list of these keywords, with no other text.

    Resume:
    {resume_text}
    """

    try:
        # Using the same free-tier model setup from previous steps
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are an expert career strategist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2  # Low temperature for factual extraction
        )

        # Extract the raw text and split it into a clean list
        raw_output = response.choices[0].message.content
        keywords = [k.strip() for k in raw_output.split(",") if k.strip()]
        return keywords

    except Exception as e:
        print(f"Error generating keywords: {e}")
        return []

def main():
    pdf_path = "resume.pdf"

    print(f"Extracting text from {pdf_path}...\n")
    resume_text = extract_resume_from_pdf(pdf_path)

    if not resume_text.strip():
        print("Failed to extract text from the PDF. It might be scanned or image-based.")
        return

    print("Asking the LLM to generate search keywords from your resume...\n")
    keywords = generate_keywords_from_resume(resume_text)

    if not keywords:
        print("Failed to generate keywords. Exiting.")
        return

    print(f"LLM Generated Keywords: {', '.join(keywords)}\n")

    if not keywords:
        print("No keywords entered. Exiting.")
        return

    print(f"\nScraping target job board for keywords: {', '.join(keywords)}...\n")
    jobs = scrape_jobs(keywords)

    if len(jobs) == 0:
        print("No jobs found matching your keywords.")
        return

    print(f"Found {len(jobs)} matching jobs. Evaluating against your resume...\n")
    print("-" * 40)

    for job in jobs:
        print(f"Role: {job['title']} at {job['company']}")
        evaluation = evaluate_job(job, resume_text)
        print(evaluation)
        print("-" * 40)


if __name__ == "__main__":
    main()