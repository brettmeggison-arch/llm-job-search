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


def scrape_jobs(url="https://realpython.github.io/fake-jobs/"):
    """
    A working web scraper using requests and BeautifulSoup.
    It fetches the HTML page, parses it, and extracts the text based on CSS classes.
    """
    print(f"Fetching jobs from {url}...")

    # 1. Make the HTTP request to get the webpage content
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)

    # 2. Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve webpage. Status code: {response.status_code}")
        return []

    # 3. Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # 4. Find all HTML elements that contain a job listing
    # (On this specific site, jobs are inside <div class="card-content">)
    job_elements = soup.find_all("div", class_="card-content")

    jobs = []
    # We only take the first 5 jobs to save your AI API credits during testing
    for element in job_elements[:5]:
        # 5. Extract specific data points from within the job container
        title_element = element.find("h2", class_="title")
        company_element = element.find("h3", class_="company")

        # We will use the location/department paragraph as the description for this sandbox site
        description_element = element.find("p", class_="location")

        # Clean up the extracted text, with fallbacks in case a field is missing
        title = title_element.text.strip() if title_element else "Unknown Title"
        company = company_element.text.strip() if company_element else "Unknown Company"
        description = description_element.text.strip() if description_element else "No description provided."

        jobs.append({
            "title": title,
            "company": company,
            "description": description
        })

    return jobs

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


def main():
    # 1. Define the path to your PDF resume
    pdf_path = "resume.pdf"  # Replace with the actual filename of your PDF

    print(f"Extracting text from {pdf_path}...\n")
    resume_text = extract_resume_from_pdf(pdf_path)

    if not resume_text.strip():
        print("Failed to extract text from the PDF. It might be scanned or image-based.")
        return

    print("Scraping target job board...\n")
    jobs = scrape_jobs()

    print(f"Found {len(jobs)} jobs. Evaluating against your resume...\n")
    print("-" * 40)

    for job in jobs:
        print(f"Role: {job['title']} at {job['company']}")
        # 2. Pass the extracted resume text into the evaluator
        evaluation = evaluate_job(job, resume_text)
        print(evaluation)
        print("-" * 40)


if __name__ == "__main__":
    main()