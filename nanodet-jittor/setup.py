#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
from setuptools import find_packages, setup

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

def get_version():
    version_file = 'nanodet/__about__.py'
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file."""
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if os.path.exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages

if __name__ == '__main__':
    import re
    
    setup(
        name='nanodet-jittor',
        version=get_version(),
        description='NanoDet-Plus implementation with Jittor framework',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='NanoDet-Jittor Contributors',
        author_email='your-email@example.com',
        keywords='computer vision, object detection, jittor',
        url='https://github.com/your-username/nanodet-jittor',
        packages=find_packages(exclude=('config', 'tools', 'demo')),
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
        license='Apache License 2.0',
        install_requires=parse_requirements('requirements.txt'),
        zip_safe=False)
