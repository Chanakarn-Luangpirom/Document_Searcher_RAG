from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
from googleapiclient.discovery import build
from keys import GOOGLE_SERVICE_ACCOUNT_FILE
import os
import time

def create_foler_in_google_drive(name):
    SCOPES = ['https://www.googleapis.com/auth/drive']
    SERVICE_ACCOUNT_FILE = GOOGLE_SERVICE_ACCOUNT_FILE

    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    
    # Create a folder
    file_metadata = {
        'name': name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    
    folder = service.files().create(body=file_metadata, fields='id').execute()
    print('Folder ID:', folder.get('id'))

create_foler_in_google_drive('Feedback')

def upload_file_to_google_drive(file_path, mime_type, parent_folder_id=None):
    SCOPES = ['https://www.googleapis.com/auth/drive']
    SERVICE_ACCOUNT_FILE = GOOGLE_SERVICE_ACCOUNT_FILE

    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)

    # File metadata
    file_metadata = {
        'name': os.path.basename(file_path),
    }
    if parent_folder_id:
        file_metadata['parents'] = [parent_folder_id]


    media = MediaFileUpload(file_path, mimetype=mime_type)

    # Upload file
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    print(f"File ID: {file.get('id')}")


def list_files_in_google_drive():
    SCOPES = ['https://www.googleapis.com/auth/drive']
    SERVICE_ACCOUNT_FILE = "ragllm-426215-7fa337710395.json"
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)

    results = service.files().list(
        pageSize=10, fields="nextPageToken, files(id, name)"
    ).execute()
    items = results.get('files', [])

    if not items:
        print('No files found.') 
    else:
        print('Files:')
        for item in items:
            print(f"{item['name']} ({item['id']})")


def delete_all_files_and_folders():
    SCOPES = ['https://www.googleapis.com/auth/drive']
    SERVICE_ACCOUNT_FILE = "ragllm-426215-7fa337710395.json"
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    page_token = None

    while True:
        response = service.files().list(q="trashed=false",
                                        spaces='drive',
                                        fields='nextPageToken, files(id, name)',
                                        pageToken=page_token).execute()

        files = response.get('files', [])
        if not files:
            print("No files found.")
            break

        for file in files:
            try:
                service.files().delete(fileId=file['id']).execute()
                print(f"Deleted file: {file['name']} (ID: {file['id']})")
                time.sleep(0.1)  # Sleep to avoid hitting the rate limit
            except Exception as e:
                print(f"An error occurred: {e}")

        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
