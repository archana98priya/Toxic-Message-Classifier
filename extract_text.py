from bs4 import BeautifulSoup
with open("scraped.html") as fp:
	soup = BeautifulSoup(fp,'html.parser')
	print(soup.get_text())
