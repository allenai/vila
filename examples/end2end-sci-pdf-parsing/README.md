# End-to-end Paper Parsing 

**Note: This tool is currently in alpha version.**
It's functional for paper parsing, but there might be potential errors or bugs. If you find any, please let us know, and we can improve them. 

## Usage 

### Docker 

It is the easiest to use our tool through docker. Basically you just need to build the docker container and start it via the following command: 

```bash 
git clone https://github.com/allenai/vila.git
cd vila/examples/end2end-sci-pdf-parsing
docker build -t vila-service .
docker run -p 8080:8080 -ti vila-service
```

And the parsing of papers can be surprising simple:

1. Parse papers through a public URL:
    ```python
    import pandas as pd 
    pdf_url = "https://arxiv.org/pdf/2106.00676.pdf" # link/to/your/paper.pdf 
    relative_coordinates = True # whether returning relative coordinates or not 
    parsed = pd.read_csv(f"http://127.0.0.1:8080/parse/?pdf_url={pdf_url}&relative_coordinates={relative_coordinates}")
    ```
2. Parse papers from local files 
    ```python
    import requests, io
    import pandas as pd 
    
    # Load local file 
    f = open("test.pdf", 'rb')
    files = {"pdf_file": (f.name, f, "multipart/form-data")}
    r = requests.post('http://localhost:8080/parse', files=files)
    parsed = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
    ```
3. Visualize the output in Python
    ```python
    import layoutparser as lp
    page_tokens, page_images = lp.load_pdf("test.pdf", load_images=True)
    for page_id in range(len(page_images)):
        cur_page_w, cur_page_h = page_images[page_id].size
        tdf = (parsed[parsed['page']==page_id][["x1", "y1", "x2", "y2"]])
        tdf['x1'] *= cur_page_w
        tdf['x2'] *= cur_page_w
        tdf['y1'] *= cur_page_h
        tdf['y2'] *= cur_page_h
        tdf = tdf.rename(columns={"x1":"x_1", "y1":"y_1", "x2":"x_2", "y2":"y_2"})
        display(
            lp.draw_box(
            page_images[page_id],
            lp.load_dataframe(tdf,block_type="rectangle")
            )
        )
    ```

The returned CSV looks like this:

|      | page | type      | text                                                                                            | block_type | block_id |      x1 |      y1 |      x2 |      y2 |
| ---: | ---: | :-------- | :---------------------------------------------------------------------------------------------- | ---------: | -------: | ------: | ------: | ------: | ------: |
|    0 |    0 | Title     | VILA: Improving Structured Content Extraction from Scientiﬁc PDFs Using ...                     |            |          |  82.806 | 70.3452 | 514.225 | 100.631 |
|    1 |    0 | Author    | Some Author ...                                                                                 |            |          | 107.183 | 117.652 | 493.351 | 174.557 |
|    2 |    0 | Abstract  | Abstract Accurately extracting structured content from PDFs is                                  |            |          |
|    3 |    0 | Section   | 1 Introduction                                                                                  |            |          |
|    4 |    0 | Paragraph | Scientiﬁc papers are usually distributed in Portable Document Format (PDF) without extensive .. |            |          |
|  ... |  ... | ...       | ...                                                                                             |            |          |
|   15 |    1 | Caption   | Figure 1: (a) Real-world scientiﬁc documents often have  ...                                    |     Figure |        5 |
|  ... |  ... | ...       | ...                                                                                             |            |          |
|   83 |    7 | Caption   | Table 2: Performance of baseline and I-VILA models on the scientiﬁc document ...                |      Table |       11 |
|  ... |  ... | ...       | ...                                                                                             |            |          |

Note:
1. If an item is a caption, it will be linked to the actual figure/caption object on that page with the corresponding `block_id`. 
2. x1, y1, x2, y2 are the left, top, right, bottom coordinate of a rectangle block.


### Command line 

Also you can access our tool through the command line, with a bit more steps: 

```bash
git clone https://github.com/allenai/vila.git
pip install -e . 
cd vila/examples/end2end-sci-pdf-parsing
python main.py --input_pdf "path/to/local/paper.pdf" 
```
