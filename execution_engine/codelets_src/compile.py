import os
import shutil
from benchmarks.model_generator import model_generator
from tools.benchmark_compilation import model_compile

def compile_model(model_name, batch, seq, init_or_gen, arch_config, half=False, gen_ORCA=None, init_ORCA=None):
    cwd = os.getcwd()
    parent = None
    # move current working directory
    codelet = os.path.join(cwd, 'codelets_src')
    if not 'codelets_src' in cwd:
        parent = cwd
        cwd = codelet
        os.chdir(codelet)
    benchmarks = os.path.join(cwd, 'benchmarks')
    tools = os.path.join(cwd, 'tools')
    arch = arch_config.replace(".json", "").replace('_','')

    # initiation phase
    if init_or_gen == 'init':

        os.chdir(benchmarks)
        model_generator(model_name, batch, seq, init_or_gen, half, gen_ORCA, init_ORCA)

        os.chdir(tools)
        model_compile(model_name, batch, seq, init_or_gen, arch_config, gen_ORCA, init_ORCA)

        os.chdir(cwd)
        if gen_ORCA == None and init_ORCA == None:
            if 'llama' in model_name:
                models = ['-embd-opt', '-rms-opt', '-rattn-opt', '-proj-opt', '-linswi-opt']
            else:
                models = ['-embd-opt', '-ln-opt', '-attn-opt', '-proj-opt', '-linear1-opt', '-linear2-opt']
        else:
            if 'llama' in model_name:
                models = ['-embd-opt', '-rms-opt', '-qkv-opt', '-proj-opt', '-linswi-opt']
            else:
                models = ['-embd-opt', '-ln-opt', '-qkv-opt', '-proj-opt', '-linear1-opt', '-linear2-opt']

        for model in models:
            model = model_name + model
            model_path = os.path.join(cwd, f"benchmarks/models/{model}.onnx")
            store_path = os.path.join(cwd, f"tools/compilation_output/{model}_{arch}_b{batch}_s{seq}_{init_or_gen}/{model}.onnx")
            shutil.move(model_path, store_path)

            if '175b' in model_name:
                if 'embd-opt' in model:
                    model_path = os.path.join(cwd, f"benchmarks/models/wte.weight")
                    store_path = os.path.join(cwd, f"tools/compilation_output/{model}_{arch}_b{batch}_s{seq}_{init_or_gen}/wte.weight")
                    shutil.move(model_path, store_path)
                    model_path = os.path.join(cwd, f"benchmarks/models/_Constant_attr__value")
                    store_path = os.path.join(cwd, f"tools/compilation_output/{model}_{arch}_b{batch}_s{seq}_{init_or_gen}/_Constant_attr__value")
                    shutil.move(model_path, store_path)
                elif 'linear1-opt' in model:
                    model_path = os.path.join(cwd, f"benchmarks/models/c_fc.bias")
                    store_path = os.path.join(cwd, f"tools/compilation_output/{model}_{arch}_b{batch}_s{seq}_{init_or_gen}/c_fc.bias")
                    shutil.move(model_path, store_path)
                    model_path = os.path.join(cwd, f"benchmarks/models/c_fc.weight")
                    store_path = os.path.join(cwd, f"tools/compilation_output/{model}_{arch}_b{batch}_s{seq}_{init_or_gen}/c_fc.weight")
                    shutil.move(model_path, store_path)
                elif 'linear2-opt' in model:
                    model_path = os.path.join(cwd, f"benchmarks/models/c_proj.bias")
                    store_path = os.path.join(cwd, f"tools/compilation_output/{model}_{arch}_b{batch}_s{seq}_{init_or_gen}/c_proj.bias")
                    shutil.move(model_path, store_path)
                    model_path = os.path.join(cwd, f"benchmarks/models/c_proj.weight")
                    store_path = os.path.join(cwd, f"tools/compilation_output/{model}_{arch}_b{batch}_s{seq}_{init_or_gen}/c_proj.weight")
                    shutil.move(model_path, store_path)

            # move to copiled_result folder
            if parent != None:
                result_folder= os.path.join(cwd, f"tools/compilation_output/{model}_{arch}_b{batch}_s{seq}_{init_or_gen}")
                if gen_ORCA == None and init_ORCA == None:
                    compiled_result = os.path.join(parent, f"compiled_result/{model_name}/b{batch}_s{seq}_{init_or_gen}/{model}_{arch}_b{batch}_s{seq}_{init_or_gen}")
                else:
                    compiled_result = os.path.join(parent, f"compiled_result/{model_name}-orca/b{batch}_s{seq}_{init_or_gen}/{model}_{arch}_b{batch}_s{seq}_{init_or_gen}")
                if os.path.exists(compiled_result):
                    shutil.rmtree(compiled_result)
                shutil.move(result_folder, compiled_result)
    # generation phase
    else:
        os.chdir(benchmarks)
        model_generator(model_name, batch, seq, init_or_gen, half, gen_ORCA, init_ORCA)

        os.chdir(tools)
        model_compile(model_name, batch, seq, init_or_gen, arch_config, gen_ORCA, init_ORCA)

        os.chdir(cwd)

        if gen_ORCA == None and init_ORCA == None:
            if 'llama' in model_name:
                models = ['-rgen-opt']
            else:
                models = ['-gen-opt']
            model_type = model_name

            for model in models:
                model = model_name + model
                model_path = os.path.join(cwd, f"benchmarks/models/{model}.onnx")
                store_path = os.path.join(cwd, f"tools/compilation_output/{model}_{arch}_b{batch}_s{seq}_{init_or_gen}/{model}.onnx")
                shutil.move(model_path, store_path)
                if parent != None:
                    result_folder= os.path.join(cwd, f"tools/compilation_output/{model}_{arch}_b{batch}_s{seq}_{init_or_gen}")
                    compiled_result = os.path.join(parent, f"compiled_result/{model_type}/b{batch}_s{seq}_{init_or_gen}/{model}_{arch}_b{batch}_s{seq}_{init_or_gen}")
                    if os.path.exists(compiled_result):
                        shutil.rmtree(compiled_result)
                    shutil.move(result_folder, compiled_result)

        else:
            models = []
            gen_len = 0
            if gen_ORCA != None:
                for i in gen_ORCA:
                    if 'llama' in model_name:
                        models.append(f'-rgen-{i}-opt')
                    else:
                        models.append(f'-gen-{i}-opt')
                    gen_len += 1
            if init_ORCA != None:
                for i in init_ORCA:
                    if 'llama' in model_name:
                        models.append(f'-rinit-{i}-opt')
                    else:
                        models.append(f'-init-{i}-opt')
            model_type = f'{model_name}-orca'
            for i, model in enumerate(models):
                model = model_name + model
                if 'gen' in model:
                    seq = gen_ORCA[i]
                    init_or_gen = 'gen'
                else:
                    seq = init_ORCA[i-gen_len]
                    init_or_gen = 'init'
                model_path = os.path.join(cwd, f"benchmarks/models/{model}.onnx")
                store_path = os.path.join(cwd, f"tools/compilation_output/{model}_{arch}_b{batch}_s{seq}_{init_or_gen}/{model}.onnx")
                shutil.move(model_path, store_path)
                if parent != None:
                    result_folder= os.path.join(cwd, f"tools/compilation_output/{model}_{arch}_b{batch}_s{seq}_{init_or_gen}")
                    compiled_result = os.path.join(parent, f"compiled_result/{model_type}/attn-{init_or_gen}/{model}_{arch}_b{batch}_s{seq}_{init_or_gen}")
                    if os.path.exists(compiled_result):
                        shutil.rmtree(compiled_result)
                    shutil.move(result_folder, compiled_result)

if __name__ == "__main__":
    compile_gpt('gpt2', 16, 256, 'init', 'benchmark_128x128.json')
    # compile_gpt('gpt2', 16, 256, 'gen', 'benchmark_128x128.json', 512)