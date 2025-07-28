import pandas as pd
import glob
import os
from datetime import date
import csv


class JobFilesMerger:
    def __init__(self, base_directory="./"):
        """
        Initialize the merger with the directory containing CSV files

        Args:
            base_directory: Directory where the CSV files are stored
        """
        self.base_directory = base_directory

        # Define all search combinations
        self.search_terms = [
            "data engineer",
            "backend engineer",
            "go",
            "golang",
            "analytics engineer",
            "python",
            "software engineer",
            "data analyst",
            "backend developer",
            "software developer",
        ]

        self.locations = [
            "Poland",
            "Portugal",
            "United Kingdom",
            "Romania",
            "Croatia",
            "EMEA",
            "Europe",
            "European Union",
            "European Economic Area",
            "Germany",
            "Serbia",
        ]

        self.job_types = ["contract", "fulltime"]

    def generate_expected_filenames(self, target_date=None):
        """
        Generate all expected filename combinations for a given date

        Args:
            target_date: Date string in format 'dd-mm-yyyy'. If None, uses today's date

        Returns:
            List of expected filenames
        """
        if target_date is None:
            target_date = date.today().strftime("%d-%m-%Y")

        expected_files = []

        for search_term in self.search_terms:
            for location in self.locations:
                for job_type in self.job_types:
                    # Clean up the values (same as in main script)
                    clean_search_term = search_term.lower().replace(" ", "_")
                    clean_location = location.lower().replace(" ", "_")
                    clean_job_type = job_type.lower().replace(" ", "_")

                    filename = f"careerflow_new_jobs_{clean_search_term}_{clean_location}_{clean_job_type}_{target_date}.csv"
                    expected_files.append(filename)

        return expected_files

    def find_existing_files(self, target_date=None):
        """
        Find all existing CSV files that match our pattern for a given date

        Args:
            target_date: Date string in format 'dd-mm-yyyy'. If None, uses today's date

        Returns:
            List of existing file paths
        """
        if target_date is None:
            target_date = date.today().strftime("%d-%m-%Y")

        # Look for files inside the date folder
        new_jobs_folder = os.path.join(self.base_directory, f"new-jobs-{target_date}")

        if not os.path.exists(new_jobs_folder):
            print(f"Folder named '{new_jobs_folder}' doesn't exist!")
            return []

        # Look for all files matching the pattern
        pattern = os.path.join(
            new_jobs_folder, f"careerflow_new_jobs_*_{target_date}.csv"
        )
        existing_files = glob.glob(pattern)

        return existing_files

    def print_search_combinations(self):
        """Print all search combinations that will be generated"""
        print("All Search Combinations:")
        print("=" * 50)

        combination_count = 0
        for search_term in self.search_terms:
            for location in self.locations:
                for job_type in self.job_types:
                    combination_count += 1
                    print(
                        f"{combination_count:3d}. {search_term} | {location} | {job_type}"
                    )

        print(f"\nTotal combinations: {combination_count}")
        return combination_count

    def merge_files(self, target_date=None, output_filename=None):
        """
        Merge all CSV files for a given date into one merged file

        Args:
            target_date: Date string in format 'dd-mm-yyyy'. If None, uses today's date
            output_filename: Custom output filename. If None, generates automatically

        Returns:
            Dictionary with merge statistics
        """
        if target_date is None:
            target_date = date.today().strftime("%d-%m-%Y")

        if output_filename is None:
            output_filename = f"careerflow_merged_jobs_{target_date}.xlsx"

        # Find all existing files
        existing_files = self.find_existing_files(target_date)

        if not existing_files:
            print(f"No CSV files found for date {target_date}")
            return {"error": "No files found", "files_processed": 0, "total_jobs": 0}

        print(f"Found {len(existing_files)} CSV files to merge:")
        for file in existing_files:
            print(f"  - {os.path.basename(file)}")

        # Read and combine all CSV files
        all_dataframes = []
        files_processed = 0
        total_jobs_before_dedup = 0

        for file_path in existing_files:
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    # Add source file info for debugging
                    df["source_file"] = os.path.basename(file_path)
                    all_dataframes.append(df)
                    files_processed += 1
                    total_jobs_before_dedup += len(df)
                    print(
                        f"  ✓ Loaded {len(df)} jobs from {os.path.basename(file_path)}"
                    )
                else:
                    print(f"  ⚠ Empty file: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  ✗ Error reading {os.path.basename(file_path)}: {e}")

        if not all_dataframes:
            print("No valid data found in any files")
            return {"error": "No valid data", "files_processed": 0, "total_jobs": 0}

        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # Remove duplicates based on Job URL (most reliable unique identifier)
        if "Job Url" in combined_df.columns:
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=["Job Url"], keep="first")
            after_dedup = len(combined_df)
            duplicates_removed = before_dedup - after_dedup
        else:
            # Fallback: use job title + company if Job Url not available
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(
                subset=["Job Title*", "Company Name*"], keep="first"
            )
            after_dedup = len(combined_df)
            duplicates_removed = before_dedup - after_dedup

        # Remove the source_file column before saving (it was just for debugging)
        if "source_file" in combined_df.columns:
            combined_df = combined_df.drop("source_file", axis=1)

        # Save the merged file
        date_folder = os.path.join(
            self.base_directory, f"new-jobs-merged-{target_date}"
        )

        # Create the folder if it doesn't exist
        os.makedirs(date_folder, exist_ok=True)

        output_path = os.path.join(date_folder, output_filename)
        combined_df.to_excel(output_path, index=False)

        # Print summary
        print(f"\n✅ Merge completed!")
        print(f"📁 Output file: {output_filename}")
        print(f"📊 Files processed: {files_processed}")
        print(f"📈 Total jobs before deduplication: {total_jobs_before_dedup}")
        print(f"🔄 Duplicates removed: {duplicates_removed}")
        print(f"📋 Final unique jobs: {after_dedup}")

        return {
            "output_file": output_filename,
            "files_processed": files_processed,
            "total_jobs_before_dedup": total_jobs_before_dedup,
            "duplicates_removed": duplicates_removed,
            "final_unique_jobs": after_dedup,
        }


# Example usage functions
def merge_todays_jobs():
    """Simple function to merge today's job files"""
    merger = JobFilesMerger()
    return merger.merge_files()


def merge_specific_date(date_string):
    """
    Merge job files for a specific date

    Args:
        date_string: Date in format 'dd-mm-yyyy' (e.g., '26-07-2025')
    """
    merger = JobFilesMerger()
    return merger.merge_files(target_date=date_string)


def show_all_combinations():
    """Show all search term combinations that the system expects"""
    merger = JobFilesMerger()
    return merger.print_search_combinations()


# Main execution
if __name__ == "__main__":
    print("Job Files Merger")
    print("=" * 40)

    # Show all combinations
    merger = JobFilesMerger()
    # total_combinations = merger.print_search_combinations()

    today = date.today().strftime("%d-%m-%Y")
    print(f"🔍 Looking for CSV files in folder: {today}/")

    # Merge today's files
    result = merger.merge_files()

    if "error" not in result:
        print(
            f"\n🎉 Successfully created merged file with {result['final_unique_jobs']} unique jobs!"
        )
    else:
        print(f"\n❌ Merge failed: {result['error']}")
        print(f"Make sure the folder '{today}/' exists with CSV files.")

# Usage examples:
#
# # Basic usage - merge today's files
# python job_merger.py
#
# # In another script:
# from job_merger import JobFilesMerger
# merger = JobFilesMerger()
# result = merger.merge_files()
#
# # Merge specific date:
# result = merger.merge_files(target_date="25-07-2025")
#
# # Show all expected combinations:
# merger.print_search_combinations()
