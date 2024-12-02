import requests
from bs4 import BeautifulSoup

def scrape_names(gender_class):
    names = []
    for key in ['a', 'b', 'c', 'c-2', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                'm', 'n', 'o', 'p', 'r', 's', 's-2', 't', 'u', 'v', 'z', 'z-2']:
        url = f'https://vardai.vlkk.lt/sarasas/{key}/'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', class_=f'names_list__links {gender_class}')
        names += [name.text for name in links]
    return names

# Vyriški ir moteriški vardai
male_names = scrape_names('names_list__links--man')
female_names = scrape_names('names_list__links--woman')

# Išsaugokite į failus
with open('vyru_vardai.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(male_names))
    
with open('moteru_vardai.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(female_names))
