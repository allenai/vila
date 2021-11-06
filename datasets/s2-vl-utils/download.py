from typing import List, Union, Dict, Any, Tuple
import sys
import zipfile
import io
import os
import hashlib
import logging
import tempfile
import shutil
from glob import glob

import requests
import pandas as pd
import requests
import layoutparser as lp
import pandas as pd
from tqdm import tqdm
from PyPDF2 import PdfFileReader, PdfFileWriter


logger = logging.getLogger(__name__)
sha1 = hashlib.sha1()
headers = {"User-Agent": "Mozilla/5.0"}

ANNOTATION_FILE = {
    "s2-vl-ver1": "https://ai2-s2-research.s3.us-west-2.amazonaws.com/s2-vlue/s2-vl-ver1-annotations.zip"
}

def bulk_fetch_pdf_for_urls(
    paper_table: pd.DataFrame,
    target_dir: str,
) -> List[List[str]]:

    os.makedirs(target_dir, exist_ok=True)
    paper_download_status = []

    paper_table = paper_table.groupby("sha").first().reset_index() # Remove duplicates
    pbar = tqdm(paper_table.iterrows(), total=len(paper_table))

    for _, row in pbar:

        sha_in_table = row["sha"]
        download_link = row["url"]

        pbar.set_description(desc=download_link)

        try:
            pdf_path = os.path.join(target_dir, str(sha_in_table) + ".pdf")

            if os.path.exists(pdf_path):
                continue

            r = requests.get(download_link, headers=headers)

            if r.status_code == 200:
                sha1.update(r.content)
                downloaded_sha = sha1.hexdigest()

                
                with open(pdf_path, "wb") as fh:
                    fh.write(r.content)

                paper_download_status.append([sha_in_table, downloaded_sha, "success"])
            else:
                print(f"Fail to download due to HTTP error {r.status_code} for {download_link}")
                paper_download_status.append([sha_in_table, None, "download_error"])
        except:
            print(f"Fail to download due to HTTP error {r.status_code} for {download_link}")
            paper_download_status.append([sha_in_table, None, "download_error"])

    return paper_download_status


def split_pdf_to_each_page_and_check(pdf_file, target_folder, remove_problematic=False):
    """Split a pdf file into separate pages.

    Args:
        pdf_file (str): The name of the PDF file to be split.
        target_folder (str): The target folder to save the splitted pages.
    """
    try:
        pdf = PdfFileReader(pdf_file)
        # Sometimes the downloaded PDF is corrupted.
        total_pages = pdf.getNumPages()
        # Sometimes some strange errors would occur if the pdf engine
        # thinks the pdf is corrupted.
    except:
        return False

    is_page_successfully_saved = []

    # Try to save individual pages
    for i in range(total_pages):
        pdf_writer = PdfFileWriter()
        pdf_writer.addPage(pdf.getPage(i))

        filename = os.path.splitext(os.path.basename(pdf_file))[0]
        save_name = os.path.join(target_folder, f"{filename}-{i:02d}.pdf")

        try:
            with open(save_name, "wb") as outputStream:
                pdf_writer.write(outputStream)
            is_page_successfully_saved.append(i)
        except KeyboardInterrupt:
            exit()
        except:
            print(f"Failed to save {save_name}")

        del pdf_writer
    del pdf

    # If individual pages
    if len(is_page_successfully_saved) != total_pages:
        is_pdf_successfully_saved = False

    else:
        ok_files = []
        saved_pdf_files = glob(f"{target_folder}/{filename}*.pdf")
        for saved_pdf_file in saved_pdf_files:
            try:
                lp.load_pdf(saved_pdf_file)
                ok_files.append(saved_pdf_file)
            except KeyboardInterrupt:
                exit()
            except:
                pass
        if len(ok_files) != total_pages:
            is_pdf_successfully_saved = False
        else:
            is_pdf_successfully_saved = True

    if not is_pdf_successfully_saved and remove_problematic:
        print(
            f"The current PDF {pdf_file} cannot be appropriately parsed. Removing the saved folders"
        )
        shutil.rmtree(target_folder)

    return is_pdf_successfully_saved


def _generalized_paper_downloading_and_processing_protocol(
    paper_table,
    target_folder,
    download_func,
):

    with tempfile.TemporaryDirectory() as combined_pdf_save_path:

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        print("Downloading the Papers")
        paper_download_status = download_func(paper_table, combined_pdf_save_path)
        paper_download_status = pd.DataFrame(
            paper_download_status, columns=["sha_in_table", "downloaded_sha", "status"]
        )

        if "page" not in paper_table.columns:
            create_download_report(paper_download_status)
            for file in glob(os.path.join(combined_pdf_save_path, "*")):
                shutil.move(file, target_folder)
            return paper_download_status

        pbar = tqdm(paper_download_status.iterrows(), total=len(paper_download_status))
        updated_paper_download_status = paper_download_status.copy()

        for idx, row in pbar:
            if row["status"] == "success":
                sha = row["sha_in_table"]
                pbar.set_description(f"Processing {sha}")

                if glob(f"{target_folder}/{sha}-*"):
                    continue  # Skip already processed

                with tempfile.TemporaryDirectory() as tempdir:
                    is_pdf_successfully_saved = split_pdf_to_each_page_and_check(
                        os.path.join(combined_pdf_save_path, sha + ".pdf"), tempdir
                    )

                    # In this command, it will save all the processed files in a tmp folder
                    # As such, when the PDFs are successfully downloaded, we need to move them
                    # to the actual target folder
                    if is_pdf_successfully_saved:
                        # The tempdir contains all the pages, but we only want to move the target pages
                        all_pages = paper_table.loc[paper_table["sha"] == sha, "page"].tolist()
                        for page in all_pages:
                            shutil.move(os.path.join(tempdir, f"{sha}-{page:02d}.pdf"), target_folder)
                    else:
                        updated_paper_download_status.iloc[idx, -1] = "pdf_parsing_failure"

        create_download_report(paper_download_status)
        return updated_paper_download_status


def create_download_report(paper_download_status):
    """Create a report of the downloaded papers.

    Args:
        paper_download_status (pd.DataFrame): The status of the downloaded papers.
    """
    print("PDF Download Report")
    incompatible_papers = paper_download_status[paper_download_status["sha_in_table"] != paper_download_status["downloaded_sha"]]

    print(f"Total mismatch: {len(incompatible_papers)}/{len(paper_download_status)}")
    print("Note: The mismatch between SHA doesn't necessarily mean\n"
          "the PDF files have different contents.")
    for _, row in incompatible_papers.iterrows():
        print(
            f"Original SHA: {row['sha_in_table']} -> Actual SHA: {row['downloaded_sha']}"
        )

    unsuccessful_papers = paper_download_status[
        paper_download_status["status"] != "success"
    ]
    for error_name, gp in unsuccessful_papers.groupby("status"):
        print(f"Total {error_name}: {len(gp)}/{len(paper_download_status)}")
        for _, row in gp.iterrows():
            print(f"Fail to download SHA: {row['sha_in_table']}")


def fetch_and_process_papers_based_on_urls(
    paper_table, target_folder
):
    return _generalized_paper_downloading_and_processing_protocol(
        paper_table, target_folder, bulk_fetch_pdf_for_urls
    )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Download S2-VL paper data")
    parser.add_argument(
        "--base-path",
        type=str,
        help="The path to the source files of a dataset, e.g., sources/s2-vl-ver1",
    )
    parser.add_argument("--annotation-table", type=str, default="annotation_table.csv")

    args = parser.parse_args()

    pdf_save_path = f"{args.base_path}/pdfs"
    if not os.path.exists(pdf_save_path):
        os.makedirs(pdf_save_path)

    paper_table = pd.read_csv(f"{args.base_path}/{args.annotation_table}")
    fetch_and_process_papers_based_on_urls(paper_table, pdf_save_path)

    print("Downloading the annotation")
    # hacky code to get the dataset name
    dataset_name = os.path.basename(args.base_path.strip("/"))
    annotation_file_url = ANNOTATION_FILE[dataset_name]

    r = requests.get(annotation_file_url)

    with zipfile.ZipFile(file=io.BytesIO(r.content)) as zip_ref:
        zip_ref.extractall(f"{args.base_path}/annotations")