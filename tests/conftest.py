import pytest
import os
import glob

@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)

@pytest.fixture(autouse=True)
def delete_npz_files():
    files_to_be_deleted = glob.glob("*.npz") + glob.glob("**/*.npz")
    for f in files_to_be_deleted:
        os.remove(f)
        print("Deleting " + f)
