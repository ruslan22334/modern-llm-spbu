"""Console for generic worker programs, tracing and debugging."""
import argparse
import json
from pathlib import Path
from emugpu import __version__, launch, export_debug_config, export_viewer


def main(argv=None):
    parser=argparse.ArgumentParser(prog='emugpu')
    parser.add_argument('--version',action='version',version=f'emugpu {__version__}')
    subs=parser.add_subparsers(dest='command',required=True)
    run=subs.add_parser('run')
    run.add_argument('entrypoint')
    run.add_argument('--world-size',type=int,default=4)
    run.add_argument('--device',default='cpu')
    run.add_argument('--devices')
    run.add_argument('--project-dir',type=Path,default=Path.cwd())
    run.add_argument('--args',default='[]'); run.add_argument('--kwargs',default='{}')
    run.add_argument('--trace'); run.add_argument('--log-dir')
    run.add_argument('--timeout',type=float,default=None); run.add_argument('--run-timeout',type=float)
    run.add_argument('--debug',action='store_true'); run.add_argument('--debug-base-port',type=int,default=5678)
    run.add_argument('--wait-for-debugger',choices=('none','rank0','all'),default='none')
    viewer=subs.add_parser('viewer'); viewer.add_argument('--output',default='trace_viewer.html'); viewer.add_argument('--force',action='store_true')
    debug=subs.add_parser('debug-config'); debug.add_argument('--world-size',type=int,default=4)
    debug.add_argument('--base-port',type=int,default=5678); debug.add_argument('--output',default='.vscode/launch.json'); debug.add_argument('--force',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='viewer': print(export_viewer(args.output,overwrite=args.force))
        elif args.command=='debug-config': print(export_debug_config(args.output,args.world_size,args.base_port,overwrite=args.force))
        else:
            positional,keyword=json.loads(args.args),json.loads(args.kwargs)
            if not isinstance(positional,list) or not isinstance(keyword,dict): raise ValueError('args=array, kwargs=object required')
            launch(args.entrypoint,world_size=args.world_size,device=args.device,
                   devices=json.loads(args.devices) if args.devices else None,
                   project_dir=args.project_dir,args=tuple(positional),kwargs=keyword,
                   trace=args.trace,log_dir=args.log_dir,timeout=args.timeout if args.timeout is not None else (600 if args.debug else 30),
                   run_timeout=args.run_timeout,debug=args.debug,debug_base_port=args.debug_base_port,
                   wait_for_debugger=args.wait_for_debugger)
    except (ValueError,TypeError,FileExistsError) as exc: parser.error(str(exc))
    return 0
