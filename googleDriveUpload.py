from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
from google.oauth2 import service_account
from googleapiclient.discovery import build
from keys import GOOGLE_SERVICE_ACCOUNT_FILE
import os
import time
import json
import io
from datetime import datetime
import pandas as pd

SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = GOOGLE_SERVICE_ACCOUNT_FILE
creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

def create_foler_in_google_drive(name):

    service = build('drive', 'v3', credentials=creds)
    
    # Create a folder
    file_metadata = {
        'name': name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    
    folder = service.files().create(body=file_metadata, fields='id').execute()
    print('Folder ID:', folder.get('id'))

def upload_file_to_google_drive(file_path, mime_type, parent_folder_id=None):

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


def upload_json_to_google_drive(json_to_upload,name,parent_folder_id = None):

    service = build('drive', 'v3', credentials=creds)

    
    json_str = json.dumps(json_to_upload)
    file_stream = io.BytesIO(json_str.encode('utf-8'))

    # File metadata
    file_metadata = {
   'name': '{}.json'.format(name),
   'mimeType': 'application/json'
    }
    
    if parent_folder_id:
        file_metadata['parents'] = [parent_folder_id]


    media = MediaIoBaseUpload(file_stream, mimetype='application/json')

    # Upload file
    file = service.files().create(
    body=file_metadata,
    media_body=media,
    fields='id'
    ).execute()
    print(f"File ID: {file.get('id')}")


def list_files_in_google_drive():

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

def list_folders_in_google_drive():

    service = build('drive', 'v3', credentials=creds)
    results = service.files().list(
        q="mimeType='application/vnd.google-apps.folder'",
        pageSize=10,  # Adjust as needed
        fields="nextPageToken, files(id, name)"
    ).execute()
    items = results.get('files', [])

    if not items:
        print('No folders found.')
    else:
        print('Folders:')
        for item in items:
            print(f"{item['name']} ({item['id']})")


def list_files_in_google_drive_folder(folder_id):
    service = build('drive', 'v3', credentials=creds)
    results = service.files().list(
        q=f"'{folder_id}' in parents",
        pageSize=10,  # Adjust as needed
        fields="nextPageToken, files(id, name)"
    ).execute()
    items = results.get('files', [])
    ids_list = []
    if not items:
        print('No files found.')
    else:
        print('Files:')
        for item in items:
            print(f"{item['name']} ({item['id']})")
            ids_list.append(item['id'])

    return ids_list


def delete_all_files_and_folders():

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

def load_json_from_google_drive(file_id):

    service = build('drive', 'v3', credentials=creds)
    try:
        response = service.files().get_media(fileId=file_id)
        
        content = response.execute()

        return content

    except Exception as e:
        print(f'An error occurred: {e}')
        return None
    
def delete_files_google_drive(file_ids = []):
    service = build('drive', 'v3', credentials=creds)
    for file_id in file_ids:
        try:
            # Delete the file
            service.files().delete(fileId=file_id).execute()
            print('File deleted successfully')
        except Exception as e:
            print('An error occurred:', e)


def show_feedback(folder_id = '1i307DYh9OFoPC6AkxI1rTwrhC0J3HKO-'):
    feedback_ids = list_files_in_google_drive_folder(folder_id)
    feedback_output = {}
    for feedback_id in feedback_ids:    
        loaded_json = load_json_from_google_drive(feedback_id)
        json_content_dict = json.loads(loaded_json.decode('utf-8'))
        print('JSON content as dictionary:')
        print(json_content_dict)
        for k,v in json_content_dict.items():
            if k in feedback_output:
                feedback_output[k].append(v)
            else:
                feedback_output[k] = [v]

    df = pd.DataFrame(feedback_output).apply(pd.Series.explode).reset_index(drop = True)
    
    return feedback_ids, df


def merge_json_feedback(folder_id = '1i307DYh9OFoPC6AkxI1rTwrhC0J3HKO-'):
    feedback_ids, df = show_feedback(folder_id)
    feedback_dict = {col: df[col].tolist() for col in df.columns}
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    try:
        upload_json_to_google_drive(feedback_dict,current_time,parent_folder_id='1i307DYh9OFoPC6AkxI1rTwrhC0J3HKO-') # Folder to store feedback
        delete_files_google_drive(feedback_ids)
    except Exception as e:
        print(f'An error occurred: {e}')

