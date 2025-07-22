import csv
import pandas as pd
import re
from datetime import date
from urllib.parse import urlparse, parse_qs
from jobspy import scrape_jobs


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

### STEP 4: Format output to match your spreadsheet schema ###
output_df = pd.DataFrame(
    {
        "Job Title*": filtered_jobs["title"],
        "Company Name*": filtered_jobs["company"],
        "Job Description": filtered_jobs["description"],
        "Job Location": filtered_jobs["location"],
        "Job Url": filtered_jobs["job_url"],
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
