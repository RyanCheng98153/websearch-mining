import re

def parse(path: str):
    parsed = []
    
    with open(path, 'r') as f:
        data = f.read()
    
        matches = re.findall(r"<num> Number: (\d+)\s+<title> (.+)", data)
        for match in matches:
            parsed.append({
                'number': match[0],
                'title': match[1]
            })
    return parsed, len(parsed)

if __name__ == '__main__':
    path = './query/trec40.txt'
    parsed, count = parse(path)
    
    for p in parsed:
        print(f"{p[0]}, {p[1]}")

    if count != 40:
        print("[Error]: count != 40")