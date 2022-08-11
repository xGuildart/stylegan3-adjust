#!/usr/bin/env python3
import sys
import os
import time
import datetime
import dropbox
import contextlib


def download(dbx, folder, subfolder, name):
    """Download a file.
    Return the bytes of the file, or None if it doesn't exist.
    """
    path = '/%s/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'), name)
    while '//' in path:
        path = path.replace('//', '/')
    with stopwatch('download'):
        try:
            md, res = dbx.files_download(path)
        except dropbox.exceptions.HttpError as err:
            print('*** HTTP error', err)
            return None
    data = res.content
    print(len(data), 'bytes; md:', md)
    return data


def upload(dbx, fullname, folder, subfolder, name, overwrite=False):
    """Upload a file.
    Return the request response, or None in case of error.
    """
    path = '/%s/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'), name)
    while '//' in path:
        path = path.replace('//', '/')
    mode = (dropbox.files.WriteMode.overwrite
            if overwrite
            else dropbox.files.WriteMode.add)
    mtime = os.path.getmtime(fullname)
    with open(fullname, 'rb') as f:
        data = f.read()
    with stopwatch('upload %d bytes' % len(data)):
        try:
            res = dbx.files_upload(
                data, path, mode,
                client_modified=datetime.datetime(*time.gmtime(mtime)[:6]),
                mute=True)
        except dropbox.exceptions.ApiError as err:
            print('*** API error', err)
            return None
    print('uploaded as', res.name.encode('utf8'))
    return res


class ddbox:
    def upload(filepath, overwrite):
        API_KEY = os.getenv('API_KEY')
        name = os.path.basename(filepath)
        if overwrite != "":
            name = overwrite
        with dropbox.Dropbox(API_KEY) as dbx:
            upload(dbx, filepath, "stylegan3", "", name, True)

    def download(filename):
        API_KEY = os.getenv('API_KEY')
        with dropbox.Dropbox(API_KEY) as dbx:
            fdata = download(dbx, "stylegan3", "", filename)
            write_to_file(filename, fdata)


'''
upload and download test
'''


def main():
    # API_KEY = os.getenv('API_KEY')

    # with dropbox.Dropbox(API_KEY) as dbx:
    #     print(dbx.users_get_current_account())
    #     print("Successfully set up client!")
    #     upload(dbx, "test_upload.txt", "stylegan3",
    #            "",  "test_upload.txt", True)
    # fdata = download(dbx, "stylegan3", "", "test_upload.txt")
    # write_to_file("test_download.txt", fdata)
    ddbox.upload("test_upload.txt")


@contextlib.contextmanager
def stopwatch(message):
    """Context manager to print how long a block of code took."""
    t0 = time.time()
    try:
        yield
    finally:
        t1 = time.time()
        print('Total elapsed time for %s: %.3f' % (message, t1 - t0))


def write_to_file(name, data):
    create_file(name)
    write_data_in_file(name, data)


def create_file(file_name):

    if sys.platform == 'linux' or sys.platform == 'darwin':
        os.system('touch ' + file_name)
    elif sys.platform == 'win32':
        os.system('echo . > ' + file_name)


def write_data_in_file(file_name, data):

    with open(file_name, 'wb') as fd:
        fd.write(data)


if __name__ == '__main__':
    main()
