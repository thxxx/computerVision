import os
import sys
import glob

# 읽어올 파일 이름
file_name = sys.argv[1]

# 폴더와 파일 생성해서 결과 작성
FILE_NAME = sys.argv[2]
f = open(os.path.join(FILE_NAME), 'w')

sha_file = open(file_name, 'r', encoding='UTF-8')
lines = sha_file.readlines()
i = 0

inst_dict = {}

for line in lines:
    tick = line[:7]
    tick = tick.strip()

    splited_line = line.split('/')

    sn = int(splited_line[1])
    op = splited_line[2]

    # 처음 들어온 sequence number라면 fetch이기 때문에 sequence number와 op먼저 해당 키에 넣어줌.
    if sn not in inst_dict.keys():
        inst_dict[sn] = []
        inst_dict[sn].append(sn)
        addsplit = splited_line[3].split(' ')
        # 아주 가끔 빈 op인 경우가 있어 그럴땐 unknown이 들어가도록 해주었다. commit 까지 끝난것만 기록하기때문에 텍스트에는 기록되지 않음.
        if addsplit[2] == '':
            inst_dict[sn].append('unknown')
        else:
            inst_dict[sn].append(addsplit[2])
        inst_dict[sn].append(tick)
    else:
        inst_dict[sn].append(tick)

for line in inst_dict.values():
    if len(line) >= 8: # commit까지 끝난 것만 출력.
        for word in line:
            f.write(f"{word}\t")
        f.write("\n")

f.close()
