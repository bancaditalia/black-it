## Hosting the paper of Black-it

Content of the paper can be found in the ``paper.md`` markdown file, and the citations used
are referenced in the ``paper.bib`` bibtex file, the format of the Journal of Open-Source Software.

For convenience, the PDF of the paper can be built locally with Docker by running the following:

```bash
sh buid_paper_docker.sh
```

which will execute the ``openjournals/paperdraft`` Docker image and output the ``paper.pdf`` file.

