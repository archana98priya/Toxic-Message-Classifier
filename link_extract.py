import requests 
def get_body(link):
    r = requests.get(link, stream = True) 
    
    with open("scraped.html","wb") as pdf: 
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: 
                pdf.write(chunk) 

    from bs4 import BeautifulSoup
    with open("scraped.html") as fp:
        soup = BeautifulSoup(fp,'html.parser')
        return soup.get_text()