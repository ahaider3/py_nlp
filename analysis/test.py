import sys
from infer import Infer

infer = Infer(sys.argv[1], 16, sys.argv[2])

text="I Hate my LIFE"

print(infer.infer(text))
