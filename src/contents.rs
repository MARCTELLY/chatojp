use std::clone::Clone;
use std::path::{Path, PathBuf};
use anyhow::Result;
use std::fs;
use crate::errors::NotAvailableError;

#[derive(Clone)]
pub struct  File {
    pub path: String,
    pub contents: String,
    pub sentences: Vec<String>,
}

impl File {
    fn new(path: &str, contents: String) -> Self {
        let sentences: Vec<String> = contents.lines().map(|line| line.to_string()).collect();
        Self {
            path: path.to_string(),
            contents,
            sentences
        }
    }
}

/// Load files from directory according to their file extension("encoding")
pub fn load_files_from_dir(dir: PathBuf, ending: &str, prefix: &PathBuf) -> Result<Vec<File>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            let mut sub_files = load_files_from_dir(path, ending, prefix)?;
            files.append(sub_files.as_mut())
        } else if path.is_file() && path.has_file_extension(ending) {
            tracing::info!("Path: {:?}", path);
            let contents = fs::read_to_string(&path)?;
            let path = Path::new(&path).strip_prefix(prefix)?.to_owned();
            let key = path.to_str().ok_or(NotAvailableError {})?;
            let file = File::new(key, contents);
            files.push(file)
        }
    }

    Ok(files)

}

trait PathBufExt {
    fn has_file_extension(&self, ending: &str) -> bool;
}

impl PathBufExt for PathBuf {
    fn has_file_extension(&self, ending: &str) -> bool {
        self.extension().map_or(false, |ext| ext == ending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File as StdFile};
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn create_test_file(dir: &PathBuf, name: &str, contents: &str) -> PathBuf {
        let file_path = dir.join(name);
        let mut file = StdFile::create(&file_path).unwrap();
        write!(file, "{}", contents).unwrap();
        file_path
    }

    #[test]
    fn test_load_files_from_dir() {
        // Create a temporary directory
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path().to_path_buf();

        // Create test files
        let file1_path = create_test_file(&temp_path, "test1.txt", "Line 1\nLine 2\nLine 3");
        let file2_path = create_test_file(&temp_path, "test2.txt", "First line\nSecond line");
        let file3_path = create_test_file(&temp_path, "test3.rs", "This is not a .txt file");

        // Define the ending and prefix
        let ending = "txt";
        let prefix = &temp_path;

        // Call the function
        let result = load_files_from_dir(temp_path.clone(), ending, prefix);

        // Verify the result
        assert!(result.is_ok());
        let files = result.unwrap();
        assert_eq!(files.len(), 2);

        let file1 = files.iter().find(|f| f.path == "test1.txt").unwrap();
        assert_eq!(file1.contents, "Line 1\nLine 2\nLine 3");
        assert_eq!(file1.sentences, vec!["Line 1", "Line 2", "Line 3"]);

        let file2 = files.iter().find(|f| f.path == "test2.txt").unwrap();
        assert_eq!(file2.contents, "First line\nSecond line");
        assert_eq!(file2.sentences, vec!["First line", "Second line"]);
    }
}

