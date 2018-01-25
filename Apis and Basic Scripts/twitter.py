from twython import Twython

t = Twython(app_key="", 
            app_secret="", 
            oauth_token="", 
            oauth_token_secret=""
            )

search = t.search(q='#PrimeStudentRep #Ad',   #**supply whatever query you want here**
                  count=100)

tweets = search['statuses']

for tweet in tweets:
  print tweet['id_str'], '\n', tweet['text'], '\n\n\n'



#   from twython import Twython, TwythonError

# # Requires Authentication as of Twitter API v1.1
# twitter = Twython(APP_KEY, APP_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
# try:
#     search_results = twitter.search(q='WebsDotCom', count=50)
# except TwythonError as e:
#     print e

# for tweet in search_results['statuses']:
#     print 'Tweet from @%s Date: %s' % (tweet['user']['screen_nam\
#                                        e'].encode('utf-8'),
#                                        tweet['created_at'])
# print tweet['text'].encode('utf-8'), '\n'