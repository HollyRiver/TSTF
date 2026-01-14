import multiprocessing as mp
import subprocess
import os
import argparse
from queue import Empty

## 1. 실험 설정
datasets = ["AIR", "coin", "ELE", "MET", "SOL", "TEM", "WID"]
model_num = 100

def should_skip(script_name, data_name):
    """
    이미 결과 파일이 존재하는지 확인하는 함수
    경로 예시: result/AIR/test/trTFTF_AIR_mse_pred.csv
    """
    script_prefix = script_name.replace(".py", "") # trTFTF.py -> trTFTF
    result_path = f"result/{data_name}/test"
    
    if os.path.exists(result_path):
        ## 해당 폴더 내의 파일 리스트를 가져옴
        files = os.listdir(result_path)

        ## 스크립트 이름으로 시작하는 파일이 하나라도 있으면 True 반환
        if any(f.startswith(script_prefix) for f in files):
            return True
        
    return False

def worker(task_queue, gpu_id):
    """
    각 GPU를 담당하는 워커. 
    큐에서 (스크립트, 데이터셋) 조합을 꺼내 shell 명령어를 실행함.
    총 21개의 조합이 존재.
    """
    while True:
        try:
            ## 큐에서 작업 하나 꺼내기
            script_name, data_name = task_queue.get(timeout=3)

            if script_name == "Scratch" or script_name == "Scratch_Original":
                lr = args.scratch_lr
            else :
                lr = args.lr

            if script_name == "Scratch_Original":
                backbone_name = "PatchTSTOriginalBackbone"
            else:
                backbone_name = "PatchTSTBackbone"
                
        except Empty:
            break

        ## 로그 파일 이름 설정 (예: log_trTFTF_coin.txt)
        os.makedirs(f"running_logs", exist_ok = True)
        log_file = f"running_logs/log_{script_name.replace('.py', '')}_{data_name}.txt"
        
        ## 실행할 명령어 구성
        cmd = [
            "python", f"{script_name}.py",
            f"--model_num={model_num}",
            f"--data={data_name}",
            f"--lr={lr}",
            f"--backbone={backbone_name}",
            "--transfer_loss=all",
            f"--device_id={gpu_id}"
        ]

        print(f"== [GPU {gpu_id}] Start: {script_name} on {data_name} ==")
        
        ## subprocess를 이용해 터미널 명령어 실행
        ## stdout/stderr를 로그 파일에 기록 (nohup 역할)
        with open(log_file, "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

        if result.returncode == 0:
            print(f"== [GPU {gpu_id}] Done: {script_name} on {data_name} ==")
        else:
            print(f"!! [GPU {gpu_id}] Failed: {script_name} on {data_name} (Check {log_file}) !!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "MultiProcessing Setting")

    parser.add_argument("--script_names", nargs = "+", help = "띄어쓰기로 .py를 제외한 실행할 스크립트 이름 입력")
    parser.add_argument("--num_gpus", type = int, default = 1, help = "GPU 개수")
    parser.add_argument("--num_process", type = int, default = 4, help = "프로세싱 개수 (GPU Utilize가 90 이상이 되도록 조정할 것)")
    parser.add_argument("--lr", type = float, default = 1e-6, help = "learning_rate")
    parser.add_argument("--scratch_lr", type = float, default = 1e-4, help = "scratch (PatchTST)")

    args = parser.parse_args()

    ## 2. 모든 작업 조합 생성 후 필터링
    all_tasks = [(s, d) for s in args.script_names for d in datasets]
    filtered_tasks = []

    print("--- 실험 상태 체크 중 ---")
    for s, d in all_tasks:
        if should_skip(s, d):
            print(f"Skipping: {s} - {d} (이미 결과가 존재합니다)")
        else:
            filtered_tasks.append((s, d))
    print(f"--- 총 {len(filtered_tasks)}개의 새로운 작업을 시작합니다 ---")
    
    ## 3. 작업 큐 생성 및 채우기
    task_queue = mp.Queue()
    for t in filtered_tasks:
        task_queue.put(t)

    ## 4. 워커 실행
    processes = []
    for i in range(args.num_process):
        p = mp.Process(target=worker, args=(task_queue, i % args.num_gpus))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("모든 실험이 종료되었습니다.")