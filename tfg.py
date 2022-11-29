import os
import json
import pypandoc
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from markdown import markdown
from sentence_transformers import SentenceTransformer

# os.system("somef describe -r https://github.com/dgarijo/Widoco/ -o test.json -t 0.8")

# https://github.com/dgarijo/Widoco/
# https://github.com/facebookresearch/encodec
# https://github.com/huggingface/transformers
# https://github.com/WongKinYiu/yolov7
# https://github.com/EleutherAI/gpt-neox
# https://github.com/ibab/tensorflow-wavenet
# https://github.com/basveeling/wavenet
# https://github.com/openai/sparse_attention
# https://github.com/jadore801120/attention-is-all-you-need-pytorch



sentences = ["I ate dinner", "I love pasta"]

def cosine(u,v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def obtenerJason (url,nombre):
  orden1="somef describe -r "
  orden2=" -o "
  orden3=" -t 0.8"
  orden=orden1+url+orden2+nombre+orden3
  print (orden)
  os.system(orden)
  return(nombre)
  
def obtenerinfo (archivo):
  f = open("test.json")
  json_load = (json.loads(f.read()))
  print(json_load["name"]["excerpt"])

def transformar(sentences,original):
  model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
  embeddings = model.encode(sentences)

  query = original
  query_vec = model.encode([query])[0]
  i=0
  for sent in sentences:
    sim = cosine(query_vec, model.encode([sent])[0])
    print("Sentence = repositorio",i, " ,similarity = ", sim)
    i+=1


def obtainReadme(archivo):
  f = open(archivo)
  json_load = (json.loads(f.read()))
  url = json_load["readmeUrl"]["excerpt"]
  print(url)
  return url

def obtainDoi(archivo):
  f = open(archivo)
  json_load = (json.loads(f.read()))
  url = json_load["citation"]["doi"]
  print(url)
  return url



urls = ["https://github.com/dgarijo/Widoco/",
 "https://github.com/facebookresearch/encodec",
 "https://github.com/huggingface/transformers",
 "https://github.com/WongKinYiu/yolov7",
 "https://github.com/ibab/tensorflow-wavenet",
 "https://github.com/basveeling/wavenet",
 "https://github.com/openai/sparse_attention",
 "https://github.com/jadore801120/attention-is-all-you-need-pytorch"]

i = 0
for url in urls:
  nombre = 'test'
  nombre2 = '.json'
  nombrecompleto=nombre+str(i)+nombre2
  print(nombrecompleto)
  #obtenerJason(url,nombrecompleto)
  i+=1

urlreadme = []
for a in range (i):
  nombre = 'test'
  nombre2 = '.json'
  nombrecompleto=nombre+str(a)+nombre2
  urlreadme.append(obtainReadme(nombrecompleto))

readmes = []
origen = "0"
j = 0
for read in urlreadme:
  #print(read)
  response = requests.get(read)
  #print(response.text)
  if j==0:
    origen=response.text
    j=1
  else:
    readmes.append(response.text)

valoresReadme = transformar(readmes,origen)

urldoi = []
for a in range (i):
  nombre = 'test'
  nombre2 = '.json'
  nombrecompleto=nombre+str(a)+nombre2
  urlreadme.append(obtainDoi(nombrecompleto))