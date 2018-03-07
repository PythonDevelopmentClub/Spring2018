import urllib
import sys

response = urllib.urlopen("https://github.com/PythonDevelopmentClub/Spring2018/blob/master/README.md")
html = response.read()
print(html)