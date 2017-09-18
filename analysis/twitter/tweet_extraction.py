def get_text(tweet):
  return tweet['text']

def get_coords(tweet):
  if 'coordinates' in tweet and tweet['coordinates']:
    return tweet['coordinates']['coordinates']

def get_place(tweet):
  if 'place' in tweet and tweet['place']:
    return tweet['place']['country']


def get_user_loc(tweet):
  if 'user' in tweet and 'location' in tweet['user']:
    return tweet['user']['location']


