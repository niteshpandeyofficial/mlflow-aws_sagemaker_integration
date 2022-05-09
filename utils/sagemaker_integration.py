from msilib.schema import MsiFileHash
import subprocess
import boto3
import json
import os
from from_root import from_root

def upload(s3_bucket_name=None,ml_run_direc=None):
    try:
        print('inside the upload function')
        output=subprocess.run(["aws","s3","sync","{}".format(ml_run_direc),"s3://{}".format(s3_bucket_name)],
                                stdout=subprocess.PIPE,encoding='utf-8')
        print('output values is ',output)
        print("\n saved to bucket:",s3_bucket_name)
        return f"done uploading :{output.stdout}"
    
    except Exception as e:
        return f"Error  occurred at the time of uploading:{e.__str__()}"

