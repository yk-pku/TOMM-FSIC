import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='multi-lable few-shot learning')
    parser.add_argument('cfg', default='configs/train.yaml', type=str)
    args = parser.parse_args()
    cfg = args.cfg
    print('cfg', cfg)

    p = 'vi_irdc'
    w = '-x ABUD-IDC1-10-56-2-41'
    ntasks = 4
    # Prepare command
    tag = cfg.replace('.yaml', '')
    num_gpus = 8 if ntasks > 8 else ntasks
    if cfg.endswith('test.yaml'):
        test = '--test'
        ntasks = 1
        num_gpus = 1
    else:
        test = ''
    if not os.path.exists('log'):
        os.mkdir('log')
    command = (
        'now=$(date +"%Y%m%d_%H%M%S")\n'
        'pwd=$(dirname $(readlink -f "$0"))\n'
        f'bk_dir=code_backup/{tag}_$now\n'
        'mkdir -p $bk_dir\n'
        'cp ./*.py $bk_dir\n'
        'cp -r datasets $bk_dir\n'
        'cp -r configs $bk_dir\n'
        'cp -r models $bk_dir\n'
        'cp -r utils $bk_dir\n'
        'OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 '
        f'srun --mpi=pmi2 -p {p} --job-name={cfg} '
        f'--gres=gpu:{num_gpus} -n{ntasks} --ntasks-per-node={num_gpus} '
        '--cpus-per-task=3 '
        f'{w} '
        '--kill-on-bad-exit=1 '
        f'python -u main.py --config $pwd/{cfg} '
        f'{test} '
        f'2>&1 | tee $pwd/log/{cfg[8 : -5]}.log-$now'
    )
    print(command)
    os.system(command)


if __name__ == '__main__':
    main()
