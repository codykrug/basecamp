
import urllib
from urllib import request
from html.parser import HTMLParser
from html.entities import name2codepoint

#open url
repo = urllib.request.urlopen('https://www.nytimes.com/2018/11/15/us/florida-recount.html?action=click&module=Top+Stories&pgtype=Homepage#commentsContainer')

#read in contents
pageContents = repo.read()

#build dict, names and comments pairs

pageContents = repo.read()


#parse data - needs work! No beautifulSoup

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start tag:", tag)
        for attr in attrs:
            print("     attr:", attr)
    def handle_endtag(self, tag):
        print("End tag  :", tag)
    def handle_data(self, data):
        print("Data     :", data)
    def handle_comment(self, data):
        print("Comment  :", data)
    def handle_entityref(self, name):
        c = chr(name2codepoint[name])
        print("Named ent:", c)
    def handle_charref(self, name):
        if name.startswith('x'):
            c = chr(int(name[1:], 16))
        else:
            c = chr(int(name))
        print("Num ent  :", c)
    def handle_decl(self, data):
        print("Decl     :", data)

parser = MyHTMLParser()
parser.feed(str(pageContents))

