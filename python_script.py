import csv
import json
import os
import pandas as pd
import re

from datetime import date
from dotenv import load_dotenv
from openai import OpenAI
from urllib.parse import urlparse, parse_qs
from jobspy import scrape_jobs

# Load environment variables
load_dotenv()

# DeepSeek API setup
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
)


def load_training_examples(excel_file_path="training_examples.xlsx"):
    """Load training examples from Excel file"""
    try:
        df = pd.read_excel(excel_file_path)

        # Validate required columns
        required_cols = ["Job Title", "Company", "Description", "Decision", "Reason"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in Excel: {missing_cols}")

        # Convert to list of examples
        examples = []
        for _, row in df.iterrows():
            examples.append(
                {
                    "title": str(row["Job Title"]),
                    "company": str(row["Company"]),
                    "description": str(row["Description"]),
                    "decision": str(row["Decision"]).upper(),
                    "reason": str(row["Reason"]),
                }
            )

        print(f"Loaded {len(examples)} training examples from {excel_file_path}")
        return examples

    except Exception as e:
        print(f"Error loading training examples: {e}")
        return []


def create_batch_job_filtering_prompt(jobs_batch, training_examples):
    """Create the prompt for DeepSeek to evaluate multiple jobs at once"""

    # Base preferences
    base_prompt = """You are a job filtering assistant. I'm a Data & Software Engineer looking for remote contract (B2B) work in Portugal. 

MY PREFERENCES:
- MUST be fully remote (no hybrid, no on-site), only quaterly (at most) travels are acceptable
- MUST NOT require data science, machine learning, or AI skills as requirements (only some basic LLM is acceptable, but it shouldn't be primary skill)
- MUST NOT require C++, Rust, PHP, Ruby, C# as primary languages, nor  require Django and Flask as primary Python framework, FastAPI is acceptable as primary framework
- I prefer Python, Go, Java only if it's not the first requiremenet (or if multiple years of experience aren't required), SQL, Docker, Kubernetes, cloud technologies
- Contract/B2B contract work preferred

IMPORTANT: If ML/data science/AI is mentioned as "nice to have" or "plus" or "preferred", that's OK - only reject if it's a hard requirement.

Here are examples of my past decisions:

"""

    # Add training examples from Excel
    examples_text = ""
    for i, example in enumerate(
        training_examples[:8]
    ):  # Use fewer examples to save tokens for multiple jobs
        examples_text += f"""EXAMPLE {i + 1}:
Title: {example["title"]}
Company: {example["company"]}
Description: {example["description"][:250]}...
DECISION: {example["decision"]}
REASON: {example["reason"]}

"""

    # Multiple jobs to evaluate
    jobs_text = "Now evaluate these jobs:\n\n"
    for i, job in enumerate(jobs_batch):
        jobs_text += f"""JOB {i + 1}:
Title: {job["title"]}
Company: {job["company"]}
Location: {job.get("location", "N/A")}
Description: {job["description"][:800]}...

"""

    response_format = f"""Respond ONLY with valid JSON array containing {len(jobs_batch)} objects:
[
  {{
    "job_id": 1,
    "decision": "KEEP" or "DISCARD",
    "reason": "Brief explanation why",
    "confidence": 1-10
  }},
  {{
    "job_id": 2,
    "decision": "KEEP" or "DISCARD", 
    "reason": "Brief explanation why",
    "confidence": 1-10
  }}
  ... (continue for all {len(jobs_batch)} jobs)
]"""

    return base_prompt + examples_text + jobs_text + response_format


def filter_jobs_batch_with_deepseek(jobs_batch, training_examples=None):
    """Use DeepSeek to evaluate multiple jobs at once"""

    if training_examples is None:
        training_examples = []

    try:
        prompt = create_batch_job_filtering_prompt(jobs_batch, training_examples)

        # Calculate approximate token count (rough estimate: 4 chars = 1 token)
        estimated_tokens = len(prompt) // 4
        print(f"Estimated tokens: {estimated_tokens}")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for consistent results
            max_tokens=500 * len(jobs_batch),  # Scale max tokens with batch size
        )

        # Parse the JSON response
        result_text = response.choices[0].message.content.strip()

        # Clean up response (sometimes models add markdown formatting)
        if result_text.startswith("```json"):
            result_text = result_text.replace("```json", "").replace("```", "")

        results = json.loads(result_text)

        # Validate we got the right number of results
        if len(results) != len(jobs_batch):
            raise ValueError(f"Expected {len(jobs_batch)} results, got {len(results)}")

        return results

    except Exception as e:
        print(f"Error processing batch: {e}")
        # Return error results for all jobs in batch
        return [
            {
                "job_id": i + 1,
                "decision": "ERROR",
                "reason": f"API error: {str(e)}",
                "confidence": 0,
            }
            for i in range(len(jobs_batch))
        ]


def apply_deepseek_filtering(
    filtered_jobs, training_file="training_examples.xlsx", batch_size=10
):
    """Apply DeepSeek filtering to the jobs dataframe using batch processing"""

    # Load training examples from Excel
    training_examples = load_training_examples(training_file)

    if not training_examples:
        print("WARNING: No training examples loaded. Using basic filtering only.")
        return filtered_jobs

    print(f"Applying DeepSeek LLM filtering with {len(training_examples)} examples...")
    print(f"Processing {len(filtered_jobs)} jobs in batches of {batch_size}")

    all_results = []

    # Process jobs in batches
    for i in range(0, len(filtered_jobs), batch_size):
        batch_end = min(i + batch_size, len(filtered_jobs))
        batch_jobs = []

        # Prepare batch data
        for index in range(i, batch_end):
            row = filtered_jobs.iloc[index]
            batch_jobs.append(
                {
                    "title": str(row["title"]),
                    "company": str(row["company"]),
                    "description": str(row["description"]),
                    "location": str(row.get("location", "")),
                }
            )

        print(
            f"Processing batch {i // batch_size + 1}/{(len(filtered_jobs) + batch_size - 1) // batch_size} ({len(batch_jobs)} jobs)..."
        )

        # Get results for this batch
        batch_results = filter_jobs_batch_with_deepseek(batch_jobs, training_examples)
        all_results.extend(batch_results)

    # Add results to dataframe
    filtered_jobs = filtered_jobs.copy()
    filtered_jobs["llm_decision"] = [r["decision"] for r in all_results]
    filtered_jobs["llm_reason"] = [r["reason"] for r in all_results]
    filtered_jobs["llm_confidence"] = [r["confidence"] for r in all_results]

    # Keep only jobs that LLM decided to keep
    final_jobs = filtered_jobs[filtered_jobs["llm_decision"] == "KEEP"].copy()

    print(f"DeepSeek filtered: {len(filtered_jobs)} → {len(final_jobs)} jobs")

    # Print some statistics
    decisions = [r["decision"] for r in all_results]
    kept = decisions.count("KEEP")
    discarded = decisions.count("DISCARD")
    errors = decisions.count("ERROR")

    print(f"Results: {kept} kept, {discarded} discarded, {errors} errors")

    # Show some examples of decisions
    print("\nSample decisions:")
    for i, result in enumerate(all_results[:3]):
        job_title = filtered_jobs.iloc[i]["title"][:50]
        print(f"  '{job_title}...' → {result['decision']}: {result['reason']}")

    return final_jobs


### STEP 1: Extract applied LinkedIn job IDs from CareerFlow ###
def extract_job_id(job_link):
    if not isinstance(job_link, str):
        return None
    if not job_link.startswith("https://www.linkedin.com"):
        return None
    # Try /jobs/view/{id}
    view_match = re.search(r"/jobs/view/(\d+)", job_link)
    if view_match:
        return view_match.group(1)
    # Try currentJobId from search URL
    parsed = urlparse(job_link)
    query_params = parse_qs(parsed.query)
    job_ids = query_params.get("currentJobId")
    if job_ids:
        return job_ids[0]
    return None


careerflow_df = pd.read_excel("./careerflow-jobs-updated.xlsx")
careerflow_df["Job Link"] = careerflow_df["Job Link"].dropna().astype(str)
applied_df = careerflow_df[
    (careerflow_df["Status"] == "Applied")
    & careerflow_df["Job Link"].str.startswith("https://www.linkedin.com")
]

applied_ids = (
    pd.Series(applied_df["Job Link"])
    .apply(extract_job_id)
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

### STEP 2: Scrape new LinkedIn jobs ###
jobs = scrape_jobs(
    site_name=[
        # "indeed",
        "linkedin",
        # "zip_recruiter",
        # "glassdoor",
        # "google",
        # "bayt",
        # "naukri",
    ],
    search_term="golang",
    is_remote=True,
    job_type="contract",
    # google_search_term="software engineer jobs near San Francisco, CA since yesterday",
    location="Portugal",
    results_wanted=20,
    hours_old=200,
    # country_indeed="USA",
    linkedin_fetch_description=True,  # gets more info such as description, direct job url (slower)
    # proxies=["208.195.175.46:65095", "208.195.175.45:65095", "localhost"],
)
print(f"Found {len(jobs)} jobs")
print(jobs.head())

### STEP 3: Filter out unwanted jobs and previously applied ones ###
jobs["title"] = jobs["title"].fillna("")
jobs["description"] = jobs["description"].fillna("")
filtered_jobs = jobs[
    # jobs["title"].str.contains("data engineer", case=False, na=False)
    ~jobs["title"].str.contains(
        r"azure|c\+\+|rust|php|ruby|spark|databricks|system|cloud engineer|devops|hybrid|on-site",
        case=False,
        na=False,
    )
    & ~jobs["description"].str.contains(
        r"c\+\+|rust|spark|pyspark|databricks|ruby|php",
        case=False,
        na=False,
    )
].copy()

print("AFTER FILTERING OUT UNWANTED JOBS")
print(filtered_jobs)

# Extract job ID from scraped job URLs
filtered_jobs["job_id"] = pd.Series(filtered_jobs["job_url"]).apply(
    lambda url: (
        match.group(1)
        if isinstance(url, str) and (match := re.search(r"/jobs/view/(\d+)", url))
        else None
    )
)

print("AFTER ADDING job_id TO THE DATA FRAME")
print(filtered_jobs)


# Exclude already-applied jobs
filtered_jobs = filtered_jobs[~pd.Series(filtered_jobs["job_id"]).isin(applied_ids)]

print("AFTER EXCLUDING ALREADY APPLIED JOBS")
print(filtered_jobs)

# NEW: Apply DeepSeek filtering
final_jobs = apply_deepseek_filtering(filtered_jobs, "training_examples.xlsx")


### STEP 4: Format output to match your spreadsheet schema ###
output_df = pd.DataFrame(
    {
        "Job Title*": final_jobs["title"],
        "Company Name*": final_jobs["company"],
        "Job Description": final_jobs["description"],
        "Job Location": final_jobs["location"],
        "Job Url": final_jobs["job_url"],
        "Tags": "",
        "Notes": "",  # placeholder for manual notes
    }
)


output_df.to_csv(
    f"careerflow_new_jobs_{date.today().strftime('%d-%m-%Y')}.csv",
    mode="a",
    quoting=csv.QUOTE_NONNUMERIC,
    escapechar="\\",
    index=False,
)  # to_excel

print(f"{len(output_df)} NEW JOBS FOUND and exported.")
