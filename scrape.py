import requests 
file_url = "https://www.azlyrics.com/lyrics/sirmixalot/babygotback.html"
  
r = requests.get(file_url, stream = True) 
  
with open("scraped.html","wb") as pdf: 
    for chunk in r.iter_content(chunk_size=1024): 
  
         # writing one chunk at a time to pdf file 
         if chunk: 
             pdf.write(chunk) 
