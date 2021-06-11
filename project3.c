
/*******************************************************************
 *       Made by : 2019147505 김호진
 *        스케줄러 메모리 관리 시뮬레이터
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <pwd.h>
#include <math.h>

#define ASSUME 30

int TimeQuantum = 10;
int pagefault = 0;

struct Process
{
    int pid;
    char *program; // char program[30] 했을 땐 안됐음.
    int pages[ASSUME];
    int pages_num;
    int priority;
    int PC;
    int aid[ASSUME];
    int startCycle;
    int sleep_rest;
    int remain_time_quantum;
    int valid[ASSUME];
    // int reference[ASSUME];
} Process;

struct IOjobs
{
    int cycle;
    int pid;
} IOjobs;

struct Memory
{
    int capacity;                // 메모리 용량
    int pageids[ASSUME];         //
    int cycle_accessed[ASSUME];  // index에 해당하는 Allocation ID를 가진 프레임이 가장 최근에 언제 access되었는지
    int now_allocated;           // 현재 할당되어있는 프레임의 개수
    int bits[257];               // 프레임이 할당되어있는 Allocation ID
    int startIndexOfAid[ASSUME]; // index에 해당하는 Allocation ID를 가진 프레임이 시작하는 bit 위치
    int lengthOfAid[ASSUME];     // index에 해당하는 Allocation ID를 가진 프레임의 길이
    int isAllocatedNow[ASSUME];  // 현재 index에 해당하는 Aid를 가진 프레임이 물리메모리를 점유하고 있는지
} Memory;

int AllocationID = 0;

int main(int argc, char *argv[])
{
    FILE *out;
    FILE *sc;
    out = fopen("memory.txt", "wt");
    sc = fopen("schedular.txt", "wt");
    char *replace_algorithm = (char *)malloc(sizeof(char) * 100);
    char *directory = (char *)malloc(sizeof(char) * 120);
    char *directory_temp = (char *)malloc(sizeof(char) * 120);
    char path[500];
    char *token = (char *)malloc(sizeof(char) * 100);
    char *tokenn = (char *)malloc(sizeof(char) * 100);
    getcwd(path, sizeof(path));

    replace_algorithm = "lru";
    directory = path;
    if (argc > 1)
    {
        char delim[] = "=";
        token = strtok(argv[1], delim);

        if (strcmp(token, "-dir") == 0)
        {
            token = strtok(NULL, delim);
            // token에는 = 이후의 키워드가 저장된다
            directory = token;
            directory = strcat(directory, "/");
        }
        else if (strcmp(token, "-page") == 0)
        {
            token = strtok(NULL, delim);
            // token에는 = 이후의 키워드가 저장된다
            replace_algorithm = token;
        }
    }
    else
    {
        directory = strcat(directory, "/");
    }
    if (argc > 2)
    {
        char delim[] = "=";
        tokenn = strtok(argv[2], delim);
        if (strcmp(tokenn, "-dir") == 0)
        {
            tokenn = strtok(NULL, delim);
            // token에는 = 이후의 키워드가 저장된다
            directory = tokenn;
            directory = strcat(directory, "/");
        }
        else if (strcmp(tokenn, "-page") == 0)
        {
            tokenn = strtok(NULL, delim);
            // token에는 = 이후의 키워드가 저장된다
            replace_algorithm = tokenn;
        }
    }

    strcpy(directory_temp, directory);
    FILE *fp = fopen(strcat(directory, "input"), "r"); // hello.txt 파일을 읽기 모드(r)로 열기.
                                                       // 파일 포인터를 반환
    strcpy(directory, directory_temp);
    char read[300] = {
        0,
    };

    fread(read, sizeof(char), 20 * 5, fp); //전체 읽기
    fclose(fp);                            // 파일 포인터 닫기

    int position = 0;
    char **tokens = malloc(sizeof(char *) * 600);

    char delim[] = " \n";
    tokens[position] = token = strtok(read, delim);
    while (token != NULL)
    {
        position++;
        token = strtok(NULL, delim);
        tokens[position] = token;
    }

    int total_event_num = atoi(tokens[0]);
    int VMsize = atoi(tokens[1]);
    int PMsize = atoi(tokens[2]);
    int page_frame_size = atoi(tokens[3]);

    struct Memory physical;
    physical.capacity = PMsize / (page_frame_size);
    physical.now_allocated = 0;

    int pid = 0;
    int io_num = 0;

    struct Process pro[total_event_num];
    struct IOjobs iojob[total_event_num];

    for (int i = 0; i < total_event_num; i++)
    {
        if (strcmp(tokens[i * 3 + 5], "INPUT") == 0)
        {
            // 이건 I/O 작업인 경우
            iojob[io_num].cycle = atoi(tokens[i * 3 + 4]);
            iojob[io_num].pid = atoi(tokens[i * 3 + 6]);
            io_num++;
        }
        else
        {
            // 이건 실행코드인 경우
            pro[pid].priority = atoi(tokens[i * 3 + 6]);
            pro[pid].pid = pid;
            pro[pid].program = tokens[i * 3 + 5];
            pro[pid].startCycle = atoi(tokens[i * 3 + 4]);
            pro[pid].PC = 0;
            pro[pid].sleep_rest = -1;
            pro[pid].pages_num = 0;
            pid++;
        }
    }

    int total_pid_num = pid;

    int cycle = 0;
    int jobs = 1;

    struct Process sleep_list[total_pid_num];
    int now_sleep_number = 0;
    for (int i = 0; i < total_pid_num; i++)
    {
        sleep_list[i].pid = -1;
    }

    struct Process IOwait_list[total_pid_num];
    int now_iowait_number = 0;
    for (int i = 0; i < total_pid_num; i++)
    {
        IOwait_list[i].pid = -1;
    }

    struct Process total_run_pros[total_pid_num];
    for (int t = 0; t < total_pid_num; t++)
    {
        total_run_pros[t].pid = -1;
    }

    struct Process now_running;
    now_running.pid = -1;
    int now_all_number_of_processes = 0; //현재 수행중인프로세스, 런큐의 프로세스수 합친

    // Run queue에는 프로세스의 pid만 넣는다.
    int RunQueue[10][10] = {
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    };

    int RunqueueLast[10] = {
        0,
    };

    while (jobs != 0)
    {
        // 한 사이클이 시작되고나서 작업들의 우선순위
        // 1
        // 1. Sleep 된 프로세스의 종료 여부 검사 -> 종료 시 Run queue 맨 뒤 삽입
        // 1
        for (int s = 0; s < total_pid_num; s++)
        {
            if (sleep_list[s].pid > -1 && sleep_list[s].pid <= total_pid_num)
            {
                sleep_list[s].sleep_rest = sleep_list[s].sleep_rest - 1;
                if (sleep_list[s].sleep_rest == 0)
                {
                    int input = RunqueueLast[sleep_list[s].priority];
                    RunQueue[sleep_list[s].priority][input] = sleep_list[s].pid;
                    RunqueueLast[sleep_list[s].priority] += 1;
                    sleep_list[s].pid = -1;
                    now_sleep_number--;
                }
            }
        }
        // 2
        // 2. Input 으로 주어진 IO작업의 시행 -> 종료 시 Run queue 맨 뒤 삽입
        // 2
        for (int i = 0; i < io_num; i++)
        {
            if (iojob[i].cycle == cycle)
            {
                for (int s = 0; s < total_pid_num; s++)
                {
                    if (IOwait_list[s].pid > -1)
                    {
                        if (IOwait_list[s].pid == iojob[i].pid)
                        {
                            // io wait list에 있는 프로세스를 빼내서 런큐에 넣는다.
                            int input = RunqueueLast[IOwait_list[s].priority];
                            RunQueue[IOwait_list[s].priority][input] = IOwait_list[s].pid;
                            RunqueueLast[IOwait_list[s].priority] += 1;
                            IOwait_list[s].pid = -1;
                            now_iowait_number--;
                            break;
                        }
                    }
                }
            }
        }

        int new_scheduled_pid = -1;
        // 3
        // 3. Input으로 주어진 프로세스 생성 작업의 시행 -> 종료 시 Run queue 맨 뒤 삽입
        // 3
        for (int p = 0; p < pid; p++)
        {
            if (cycle == pro[p].startCycle)
            {
                // 프로세스 생성.
                // 현재 프로세스 목록에 추가
                // 런 큐 맨뒤에 추가 PC는 0으로 해서

                if (now_running.pid == -1) // 첫번째면 바로 실행한다.
                {
                    now_running = pro[p];
                    TimeQuantum = 10;
                    new_scheduled_pid = now_running.pid;
                }
                else
                {
                    int input = RunqueueLast[pro[p].priority];
                    RunQueue[pro[p].priority][input] = pro[p].pid; // 맨뒤로 수정
                    RunqueueLast[pro[p].priority] += 1;
                }
                total_run_pros[pro[p].pid] = pro[p];
                now_all_number_of_processes += 1;
            }
        }
        // 4. 이번 사이클에 실행될 Process 결정
        // 런 큐 안에서 제일 우선순위 높은거 결정 -> 현재 돌고 있는거랑 비교
        // 원래 프로세스가 뺏길 시 그때의 PC를 저장해둬야함.
        int priority = 10;
        for (int r = 0; r < 10; r++)
        {
            if (RunQueue[r][0] > -1)
            {
                // RunQueue[r]의 priority 저장
                priority = r;
                // 지금 런큐에서 가장 작은 priority
                break;
            }
        }

        fprintf(sc, "[%d Cycle] Scheduled Process: ", cycle);
        // now_running 0-pid, 1-priority, 2-program명, 3-PC
        // 원래 실행중이던 process가 종료 혹은 block되어서 나간 경우 바로 아까 그 런큐의 프로세스를 실행한다.
        if (now_running.pid == -1)
        {
            if (priority != 10)
            {
                // 새롭게 now_running 부여
                now_running = pro[RunQueue[priority][0]];
                TimeQuantum = 10;
                // 런 큐에서 프로세스 빼서 없앰 그럼 한칸씩 앞으로 당겨야한다.
                RunQueue[priority][0] = -1;

                // 당기는 과정
                int check_compact = 1;
                while (RunQueue[priority][check_compact] != -1)
                {
                    RunQueue[priority][check_compact - 1] = RunQueue[priority][check_compact];
                    check_compact += 1;
                }
                RunQueue[priority][check_compact - 1] = -1;
                RunqueueLast[priority] -= 1;
                // 당기는 과정

                new_scheduled_pid = now_running.pid;
            }
        }
        else if (now_running.priority > priority)
        {
            // now_running의 정보들은 런큐 맨 뒤에 들어감
            int input = RunqueueLast[now_running.priority];
            RunQueue[now_running.priority][input] = now_running.pid;
            RunqueueLast[now_running.priority] += 1;

            pro[now_running.pid] = now_running;
            // 새롭게 now_running 부여
            now_running = pro[RunQueue[priority][0]];
            TimeQuantum = 10;
            // 런 큐에서 프로세스 빼서 없앰
            RunQueue[priority][0] = -1;

            // 당기는 과정 priority변수를 인자로 받는 함수로 모듈화하기
            int check_compact = 1;
            while (RunQueue[priority][check_compact] != -1)
            {
                RunQueue[priority][check_compact - 1] = RunQueue[priority][check_compact];
                check_compact += 1;
            }
            RunQueue[priority][check_compact - 1] = -1;
            RunqueueLast[priority] -= 1;
            // 당기는 과정

            new_scheduled_pid = now_running.pid;
            //
        }

        if (new_scheduled_pid > -1)
        {
            fprintf(sc, "%d %s (priority %d)\n", now_running.pid, now_running.program, now_running.priority);
        }
        else
        {
            fprintf(sc, "None\n");
        }
        if (now_running.pid > -1)
        {
            if (now_running.priority <= 9 && 5 <= now_running.priority)
            {
                TimeQuantum -= 1;
            }
            // 5. 이번 사이클에 실행된 Process의 명령을 확인하여 명령 실행
            // now_process의 program명으로 파일 열어서 PC위치로가서 숫자 읽어서 조건문으로 실행한다.
            strcpy(directory_temp, directory);
            FILE *fp = fopen(strcat(directory, now_running.program), "r"); // 읽기 모드(r)로 열기.
                                                                           // 파일 포인터를 반환
            strcpy(directory, directory_temp);

            char read[300] = {
                0,
            };

            fread(read, sizeof(char), 20 * 5, fp); //전체 읽기
            fclose(fp);                            // 파일 포인터 닫기

            int position = 0;
            char **tokens = malloc(sizeof(char *) * 600);

            char delim[] = " \n";
            tokens[position] = token = strtok(read, delim);
            while (token != NULL)
            {
                position++;
                token = strtok(NULL, delim);
                tokens[position] = token;
            }
            // Line 2
            fprintf(sc, "Running Process: ");

            fprintf(sc, "Process#%d(%d) running code %s line %d(op %d, arg %d)\n", now_running.pid, now_running.priority, now_running.program,
                    now_running.PC + 1, atoi(tokens[now_running.PC * 2 + 1]), atoi(tokens[now_running.PC * 2 + 2]));

            // 페이지 테이블에 연속된 페이지 개수만큼 페이지 할당
            if (atoi(tokens[now_running.PC * 2 + 1]) == 0)
            {
                int required_pages = atoi(tokens[now_running.PC * 2 + 2]);
                fprintf(out, "[%d Cycle] Input: Pid[%d] Function[%s] Page ID[%d] Page Num[%d]\n", cycle, now_running.pid, "ALLOCATION", now_running.pages_num, required_pages);
                now_running.pages[now_running.pages_num] = required_pages;
                now_running.valid[now_running.pages_num] = 0;
                now_running.aid[now_running.pages_num] = -1;
                // now_running.reference[now_running.pages_num] = -1;
                now_running.pages_num++;
            }
            else if (atoi(tokens[now_running.PC * 2 + 1]) == 1)
            {
                // page ID로 접근을 요청한다. process구분없이 전체에 대하여 고유한 AID를 부여받고, valid는 1이된다.
                int accesspid = atoi(tokens[now_running.PC * 2 + 2]);
                fprintf(out, "[%d Cycle] Input: Pid[%d] Function[%s] Page ID[%d] Page Num[%d]\n", cycle, now_running.pid, "ACCESS", accesspid, now_running.pages[accesspid]);
                // 새로운 Allocation ID 할당.
                if (now_running.aid[accesspid] <= -1)
                {
                    pagefault += 1;

                    now_running.aid[accesspid] = AllocationID;
                    for (int i = 0; i < now_running.pages_num; i++)
                    {
                        now_running.valid[i] = 0;
                    }
                    now_running.valid[accesspid] = 1;
                    AllocationID += 1;

                    for (int r = 1; r < 6; r++)
                    {
                        if (pow(2, r - 1) < now_running.pages[accesspid] && now_running.pages[accesspid] < pow(2, r))
                        {
                            while (physical.capacity - physical.now_allocated - pow(2, r) < 0)
                            {
                                // 페이지 교체 알고리즘을 실행해야함.
                                if (strcmp(replace_algorithm, "lru") == 0)
                                {
                                    // 가장 예전에 사용된 프레임을 골라서 해제한다.
                                    int lruframe = 300; // 그냥 작은수로 잡은 것
                                    int removeAid = -1;
                                    for (int c = 0; c < ASSUME; c++)
                                    {
                                        if (physical.isAllocatedNow[c] > 0)
                                        {
                                            if (0 < physical.cycle_accessed[c] && physical.cycle_accessed[c] < lruframe)
                                            {
                                                removeAid = c;
                                            }
                                        }
                                    }
                                    // removeAid를 메모리 해제한다.
                                    for (int c = physical.startIndexOfAid[removeAid]; c < physical.startIndexOfAid[removeAid] + physical.lengthOfAid[removeAid]; c++)
                                    {
                                        physical.bits[c] = -1;
                                    }
                                    physical.isAllocatedNow[removeAid] = 0;
                                    physical.now_allocated -= physical.lengthOfAid[removeAid];
                                    // 함수로 따로 빼기 정리할 시간이 있다면..
                                }
                                else if (strcmp(replace_algorithm, "sampled") == 0)
                                {
                                    // 가장 예전에 사용된 프레임을 골라서 해제한다.
                                    int lruframe = 300; // 그냥 작은수로 잡은 것
                                    int removeAid = -1;
                                    for (int c = 0; c < ASSUME; c++)
                                    {
                                        if (physical.isAllocatedNow[c] > 0)
                                        {
                                            if (0 < physical.cycle_accessed[c] && physical.cycle_accessed[c] < lruframe)
                                            {
                                                removeAid = c;
                                            }
                                        }
                                    }
                                    // removeAid를 메모리 해제한다.
                                    for (int c = physical.startIndexOfAid[removeAid]; c < physical.startIndexOfAid[removeAid] + physical.lengthOfAid[removeAid]; c++)
                                    {
                                        physical.bits[c] = -1;
                                    }
                                    physical.isAllocatedNow[removeAid] = 0;
                                    physical.now_allocated -= physical.lengthOfAid[removeAid];
                                    // 함수로 따로 빼기 정리할 시간이 있다면..
                                }
                                else if (strcmp(replace_algorithm, "clock") == 0)
                                {
                                    // 가장 예전에 사용된 프레임을 골라서 해제한다.
                                    int lruframe = 300; // 그냥 작은수로 잡은 것
                                    int removeAid = -1;
                                    for (int c = 0; c < ASSUME; c++)
                                    {
                                        if (physical.isAllocatedNow[c] > 0)
                                        {
                                            if (0 < physical.cycle_accessed[c] && physical.cycle_accessed[c] < lruframe)
                                            {
                                                removeAid = c;
                                            }
                                        }
                                    }
                                    // removeAid를 메모리 해제한다.
                                    for (int c = physical.startIndexOfAid[removeAid]; c < physical.startIndexOfAid[removeAid] + physical.lengthOfAid[removeAid]; c++)
                                    {
                                        physical.bits[c] = -1;
                                    }
                                    physical.isAllocatedNow[removeAid] = 0;
                                    physical.now_allocated -= physical.lengthOfAid[removeAid];
                                    // 함수로 따로 빼기 정리할 시간이 있다면..
                                }
                                else
                                {
                                    printf("What do you want? \n");
                                    break;
                                }
                            }

                            for (int al = physical.now_allocated; al < physical.now_allocated + pow(2, r); al++)
                            {
                                physical.bits[al] = now_running.aid[accesspid];
                            }
                            physical.startIndexOfAid[now_running.aid[accesspid]] = physical.now_allocated;
                            physical.now_allocated += pow(2, r);

                            physical.cycle_accessed[now_running.aid[accesspid]] = cycle;
                            physical.isAllocatedNow[now_running.aid[accesspid]] = 1;
                            physical.lengthOfAid[now_running.aid[accesspid]] = pow(2, r);
                            break;
                        }
                    }
                }
                else if (physical.isAllocatedNow[now_running.aid[accesspid]] <= 0)
                {
                    pagefault += 1;
                }
            }
            else if (atoi(tokens[now_running.PC * 2 + 1]) == 2)
            {
                fprintf(out, "[%d Cycle] Input: Pid[%d] Function[%s] Page ID[%d] Page Num[%d]\n", cycle, now_running.pid, "RELEASE", 1, 1);
                int releasepid = atoi(tokens[now_running.PC * 2 + 2]);
                int removeAid = now_running.aid[releasepid];
                // removeAid를 메모리 해제한다.
                for (int c = physical.startIndexOfAid[removeAid]; c < physical.startIndexOfAid[removeAid] + physical.lengthOfAid[removeAid]; c++)
                {
                    physical.bits[c] = -1;
                }
                physical.isAllocatedNow[removeAid] = 0;
                physical.now_allocated -= physical.lengthOfAid[removeAid];
                // 함수로 따로 빼기 정리할 시간이 있다면..
            }
            else if (atoi(tokens[now_running.PC * 2 + 1]) == 3)
            {
                fprintf(out, "[%d Cycle] Input: Pid[%d] Function[%s]\n", cycle, now_running.pid, "NON-MEMORY");
            }
            else if (atoi(tokens[now_running.PC * 2 + 1]) == 4)
            {
                fprintf(out, "[%d Cycle] Input: Pid[%d] Function[%s]\n", cycle, now_running.pid, "SLEEP");
                now_running.sleep_rest = atoi(tokens[now_running.PC * 2 + 2]);
                now_running.PC = now_running.PC + 1;
                pro[now_running.pid] = now_running;
                sleep_list[now_running.pid] = now_running;
                now_sleep_number++;
                now_running.pid = -1;
            }
            else if (atoi(tokens[now_running.PC * 2 + 1]) == 5)
            {
                fprintf(out, "[%d Cycle] Input: Pid[%d] Function[%s]\n", cycle, now_running.pid, "IOWAIT");
                now_running.PC = now_running.PC + 1;
                pro[now_running.pid] = now_running;
                IOwait_list[now_running.pid] = now_running;
                now_running.pid = -1;
                now_iowait_number++;
            }
            if (now_running.pid != -1)
            {
                now_running.PC = now_running.PC + 1;
            }
            if (now_running.PC == atoi(tokens[0]))
            {
                now_all_number_of_processes -= 1;
                total_run_pros[now_running.pid].pid = -1;
                now_running.pid = -1;
                // 해당 프로세스는 모든 명령을 완료하였다. physical memory에서 해당 프로세스에게 할당되어 있는 프레임들도 해제되어야한다.
                for (int ai = 0; ai < ASSUME; ai++)
                {
                    if (physical.isAllocatedNow[ai] == 1)
                    {
                        int removeAid = ai;

                        for (int c = physical.startIndexOfAid[removeAid]; c < physical.startIndexOfAid[removeAid] + physical.lengthOfAid[removeAid]; c++)
                        {
                            physical.bits[c] = -1;
                        }
                        physical.isAllocatedNow[removeAid] = 0;
                        physical.now_allocated -= physical.lengthOfAid[removeAid];
                    }
                }
                //
            }
        }
        else
        {
            fprintf(sc, "Running Process: None \n");
        }
        if (now_running.pid != -1)
        {
            total_run_pros[now_running.pid] = now_running;
        }

        fprintf(out, "%-30s", ">> Physical Memory:");
        int pm_nums = PMsize / (page_frame_size);
        for (int pm = 0; pm < pm_nums; pm++)
        {
            if ((pm % 4) == 0)
            {
                fprintf(out, "|");
            }
            if (physical.bits[pm] > -1 && physical.bits[pm] < 100)
            {
                fprintf(out, "%d", physical.bits[pm]);
            }
            else
            {
                fprintf(out, "-");
            }
        }
        fprintf(out, "|\n");

        // virtual memory 출력

        int vm_nums = VMsize / (page_frame_size);
        for (int i = 0; i < total_pid_num; i++) // 1을 들어와서 아직 종료되지 않은 프로세스 개수로 바꿔야한다.
        {
            if (total_run_pros[i].pid > -1 && total_run_pros[i].pid < total_pid_num)
            {
                fprintf(out, ">> pid(%d)%-20s", total_run_pros[i].pid, " Page Table(PID): ");
                // pid에 -를 출력할지 process id를 출력할지 결정
                for (int vm = 0; vm < vm_nums; vm++)
                {
                    if (vm % 4 == 0)
                    {
                        fprintf(out, "|");
                    }
                    int check = -1;
                    int draw = 0;
                    for (int p = 0; p < total_run_pros[i].pages_num; p++)
                    {
                        if (check < vm && vm <= total_run_pros[i].pages[p] + check)
                        {
                            fprintf(out, "%d", p);
                            draw = 1;
                            break;
                        }
                        check += total_run_pros[i].pages[p];
                    }
                    if (draw == 0)
                    {
                        fprintf(out, "-");
                    }
                }
                fprintf(out, "|\n");
                //
                // allocation ID를 출력하는 부분
                //
                fprintf(out, ">> pid(%d)%-20s", total_run_pros[i].pid, " Page Table(AID): ");
                for (int vm = 0; vm < vm_nums; vm++)
                {
                    if (vm % 4 == 0)
                    {
                        fprintf(out, "|");
                    }
                    int check = -1;
                    int draw = 0;
                    for (int p = 0; p < total_run_pros[i].pages_num; p++)
                    {
                        if (check < vm && vm <= total_run_pros[i].pages[p] + check)
                        {
                            if (total_run_pros[i].aid[p] > -1)
                            {
                                fprintf(out, "%d", total_run_pros[i].aid[p]);
                                draw = 1;
                                break;
                            }
                            else
                            {
                                break;
                            }
                        }
                        check += total_run_pros[i].pages[p];
                    }
                    if (draw == 0)
                    {
                        fprintf(out, "-");
                    }
                }
                fprintf(out, "|\n");
                //
                // valid bit을 출력하는 부분
                //
                fprintf(out, ">> pid(%d)%-20s", total_run_pros[i].pid, " Page Table(Valid): ");
                for (int vm = 0; vm < vm_nums; vm++)
                {
                    if (vm % 4 == 0)
                    {
                        fprintf(out, "|");
                    }
                    int check = -1;
                    int draw = 0;
                    for (int p = 0; p < total_run_pros[i].pages_num; p++)
                    {
                        if (check < vm && vm <= total_run_pros[i].pages[p] + check)
                        {
                            fprintf(out, "%d", total_run_pros[i].valid[p]);
                            draw = 1;
                            break;
                        }
                        check += total_run_pros[i].pages[p];
                    }
                    if (draw == 0)
                    {
                        fprintf(out, "-");
                    }
                }
                fprintf(out, "|\n");
                //
                fprintf(out, ">> pid(%d)%-20s", total_run_pros[i].pid, " Page Table(Ref): ");
                for (int vm = 0; vm < vm_nums; vm++)
                {
                    if (vm % 4 == 0)
                    {
                        fprintf(out, "|");
                    }
                    fprintf(out, "-");
                }
                fprintf(out, "|\n");
            }
        }

        // 6. 텍스트 파일에 정보 출력
        if (TimeQuantum <= 0)
        {
            int input = RunqueueLast[now_running.priority];
            RunQueue[now_running.priority][input] = now_running.pid;
            RunqueueLast[now_running.priority] += 1;
            pro[now_running.pid] = now_running;
            now_running.pid = -1;
        }

        // Line 3

        for (int priority = 0; priority < 10; priority++)
        {
            fprintf(sc, "RunQueue %d:", priority);
            if (RunQueue[priority][0] == -1)
            {
                fprintf(sc, " Empty");
            }
            else
            {
                int r = 0;
                while (RunQueue[priority][r] > -1)
                {
                    for (int s = 0; s < total_pid_num; s++)
                    {
                        if (pro[s].pid == RunQueue[priority][r])
                        {
                            fprintf(sc, " %d(%s)", RunQueue[priority][r], pro[s].program);
                        }
                    }
                    r++;
                }
            }
            fprintf(sc, "\n");
        }

        //슬립 리스트 출력
        fprintf(sc, "SleepList: ");
        if (now_sleep_number == 0)
        {
            fprintf(sc, "Empty");
        }
        else
        {
            for (int s = 0; s < total_pid_num; s++)
            {
                if (sleep_list[s].pid > -1)
                {
                    if (sleep_list[s].sleep_rest > 0)
                    {
                        fprintf(sc, "%d(%s) ", sleep_list[s].pid, sleep_list[s].program);
                    }
                }
            }
        }
        fprintf(sc, "\n");
        fprintf(out, "\n");

        // 아이오 웨이트 리스트 출력
        fprintf(sc, "IOwait List: ");
        if (now_iowait_number == 0)
        {
            fprintf(sc, "Empty");
        }
        else
        {
            for (int s = 0; s < total_pid_num; s++)
            {
                if (IOwait_list[s].pid > -1 && IOwait_list[s].pid < total_pid_num)
                {
                    fprintf(sc, "%d(%s)", IOwait_list[s].pid, IOwait_list[s].program);
                }
            }
        }
        fprintf(sc, "\n");
        fprintf(sc, "\n");

        if (now_all_number_of_processes <= 0)
        {
            jobs = 0;
        }
        cycle++;
    }

    fprintf(out, "page fault : %d \n", pagefault);
    return 0;
}
