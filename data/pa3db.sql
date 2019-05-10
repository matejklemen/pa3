CREATE TABLE IndexWord (
  word TEXT PRIMARY KEY
);

CREATE TABLE Posting (
  word TEXT NOT NULL,
  documentName TEXT NOT NULL,
  frequency INTEGER NOT NULL,
  indexes TEXT NOT NULL,
  PRIMARY KEY(word, documentName),
  FOREIGN KEY (word) REFERENCES IndexWord(word)
);
