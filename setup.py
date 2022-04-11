import torch
import glob
import os
import setuptools
import sys
import torch.utils.cpp_extension


sys.path.append(os.path.dirname(__file__))

EXTENSIONS = []
CMD_CLASS = {}

def add_cpp_extension():
    extra_compile_args = [
        '-std=c++17' if not sys.platform.startswith('win') else '/std:c++17',
    ]
    extra_link_args = []
    define_macros = [
        ('_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS', None),  # mostly for the pytorch codebase
    ]

    if sys.platform.startswith('win'):
        extra_compile_args += ['/permissive']
        define_macros += [('OPENPIFPAF_FURNITURE_DLLEXPORT', None)]

    if os.getenv('DEBUG', '0') == '1':
        print('DEBUG mode')
        if sys.platform.startswith('linux'):
            extra_compile_args += ['-g', '-Og']
            extra_compile_args += [
                '-Wuninitialized',
                # '-Werror',  # fails in pytorch code, but would be nice to have in CI
            ]
        define_macros += [('DEBUG', None)]

    this_dir = os.path.dirname(os.path.abspath(__file__))
    EXTENSIONS.append(
        torch.utils.cpp_extension.CppExtension(
            'openpifpaf_furniture._cpp',
            glob.glob(os.path.join(this_dir, 'openpifpaf_furniture', 'csrc', 'src', '**', '*.cpp'), recursive=True),
            depends=glob.glob(os.path.join(this_dir, 'openpifpaf_furniture', 'csrc', 'include', '**', '*.hpp'), recursive=True),
            include_dirs=[os.path.join(this_dir, 'openpifpaf_furniture', 'csrc', 'include')],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    )
    #assert 'build_ext' not in CMD_CLASS
    CMD_CLASS['build_ext'] = torch.utils.cpp_extension.BuildExtension.with_options(no_python_abi_suffix=True)


add_cpp_extension()
setuptools.setup(
    name='openpifpaf_furniture',
    version="0.1.0",
    description='OpenPifPaf for furniture detection',
    packages=setuptools.find_packages(),
    package_data={
        'openpifpaf_furniture': ['*.dll', '*.dylib', '*.so'],
    },
    cmdclass=CMD_CLASS,
    ext_modules=EXTENSIONS,
    zip_safe=False,

    python_requires='>=3.6',
)
