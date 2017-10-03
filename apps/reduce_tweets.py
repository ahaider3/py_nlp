import analysis


data = list(analysis.read("tweets"))
s = set()
new_data = []
for d in data:
  if d['old_title'] not in s:
    s.add(d['old_title'])
    new_data.append(d)
analysis.delete("tweets_1")
analysis.write(new_data, "tweets_1")
