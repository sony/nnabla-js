#!/usr/bin/env python3
# Copyright 2022 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import pathlib
import re
import subprocess
from tqdm import tqdm
from contextlib import contextmanager

_exclude_dirs = ['.git', '.egg', '.vscode',
                 'build', 'dist', 'node_modules', '.npm'
                 ]
_exclude_files = [".gitignore", "LICENSE", "NOTICE", ".DS_Store", ".config"]
_date_extract_regex = re.compile('(\\d{4}-\\d{2}-\\d{2})')
_shebang = re.compile('#!.+')
_exclude_regex = re.compile('update copyright')


def execute_command(command):
    return subprocess.check_output(command).decode("utf-8")


def retrieve_commit_dates(filepath):
    current_dir = os.getcwd()
    parent = str(filepath.parent)
    os.chdir(parent)
    result = execute_command(
        ['git', 'log', '--format="%ci,%s"', '--reverse', filepath.name])
    os.chdir(current_dir)
    excluded = ""
    for line in result.split('\n'):
        if _exclude_regex.search(line):
            continue
        else:
            excluded += line
    dates = _date_extract_regex.findall(excluded)
    return dates


def fill_intermediate_years(years, end_year=None):
    years.sort()
    start_year = int(years[0])
    if end_year is None:
        end_year = int(years[-1])
    filled_years = []
    for year in range(start_year, end_year+1):
        filled_years.append(str(year))
    return filled_years


class Checker:
    def __init__(self, postfix_list, prefix, type):
        self.prefix = prefix
        self.shebang = re.compile('#!.+')
        self.copyright_sony_corp = self.prefix + ' Copyright {} Sony Corporation.'
        self.copyright_sony_group_corp = self.prefix + \
            ' Copyright {} Sony Group Corporation.'
        if type == "html":
            self.header_extract_rule = re.compile(
                '^<!--[\\n]{1,3} Copyright.*? limitations under the License.\n-->\n', re.DOTALL)
        else:
            self.header_extract_rule = re.compile(
                '[^\\n]{1,3} Copyright.*? limitations under the License.\n', re.DOTALL)
        apache2_license_template = '''#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
        if prefix != '#':
            self.apache2_license_template = apache2_license_template.replace(
                "#", self.prefix)
        else:
            self.apache2_license_template = apache2_license_template
        self.postfix_list = postfix_list
        self.text = None
        self.type = type

    def extract_shebang(self):
        print(self.text)
        extracted_texts = _shebang.findall(self.text)
        if len(extracted_texts) == 0:
            return None
        return extracted_texts[0]

    def has_shebang(self):
        try:
            has_she_bang = self.extract_shebang() is not None
            return has_she_bang
        except UnicodeDecodeError:
            # Raw binary file and not text.
            return False

    def accept_type(self, f):
        fn = str(f)
        base_name = os.path.basename(fn)
        if base_name in _exclude_files:
            return False
        if base_name in self.postfix_list:
            return True
        body, ext = os.path.splitext(base_name)
        if ext in _exclude_files:
            return False
        if body in _exclude_files:
            return False
        if body in self.postfix_list:
            return True
        if ext == ".tmpl":
            fn = fn.replace(".tmpl", "")
            ext = os.path.splitext(fn)[1]
        return ext in self.postfix_list

    @contextmanager
    def read_file(self, fn):
        try:
            with open(fn, "r", encoding='utf-8') as f:
                self.text = f.read()
            yield self.text
            self.text = None
        except UnicodeDecodeError:
            print("{} is invalid text file, skipped!".format(fn))
            yield None

    def create_file_header(self, f):
        commit_dates = retrieve_commit_dates(f)
        if len(commit_dates) == 0:
            return None

        sony_years = set()
        sony_group_years = set()
        for date in commit_dates:
            (year, month, _) = date.split('-')
            if int(year) <= 2020:
                sony_years.add(year)
            elif int(year) <= 2021 and int(month) < 4:
                sony_years.add(year)
            else:
                sony_group_years.add(year)

        if self.type == 'html':
            header = '<!--\n'
        else:
            header = ''
        if len(sony_years) != 0:
            sony_years = list(sony_years)
            sony_years = fill_intermediate_years(sony_years, end_year=2021)
            joined_sony_years = ','.join(sony_years)
            header += self.copyright_sony_corp.format(joined_sony_years) + '\n'

        if len(sony_group_years) != 0:
            sony_group_years = list(sony_group_years)
            sony_group_years = fill_intermediate_years(sony_group_years)
            joined_sony_group_years = ','.join(sony_group_years)
            header += self.copyright_sony_group_corp.format(
                joined_sony_group_years) + '\n'

        header += self.apache2_license_template

        if self.type == 'html':
            header += "-->\n"

        return header

    def extract_file_header(self):
        extract_texts = self.header_extract_rule.findall(self.text)
        if len(extract_texts) == 0:
            return None
        return extract_texts[0]

    def replace_file_header(self, old_header, new_header):
        if old_header is None:
            she_bang = self.extract_shebang()
            if she_bang is None:
                replaced_text = new_header + '\n' + self.text
            else:
                text = self.text.replace(she_bang + '\n', '')
                replaced_text = she_bang + '\n' + new_header + '\n' + text
        else:
            replaced_text = self.text.replace(old_header, new_header)

        return replaced_text


def list_up_files(root_dir, checkers):
    files = []
    path = pathlib.Path(root_dir)
    dir_name = str(path)
    if os.path.basename(dir_name) in _exclude_dirs:
        return files

    for f in path.iterdir():
        if f.is_dir():
            files.extend(list_up_files(f, checkers))
            continue

        for type, c in checkers.items():
            if c.accept_type(str(f)):
                files.append((f, type))
                break

    return files


def main(args):
    types = {
        "script": (
            [".py", ".sh", "Dockerfile"],
            "#"
        ),
        "js": (
            [".js", ".ts"],
            "//"
        ),
        "html": (
            [".html"],
            "",
        )
    }
    checkers = {}
    for k, v in types.items():
        checkers[k] = Checker(*v, type=k)

    # list of (file_path, type)
    files = list_up_files(args.rootdir, checkers)

    for fn, type in tqdm(files):
        c = checkers[type]
        new_header = c.create_file_header(fn)
        if new_header is None:
            continue

        with c.read_file(str(fn)) as f:
            if not f:
                continue

            old_header = c.extract_file_header()
            if new_header == old_header:
                continue
            if args.check_only:
                raise ValueError(
                    f"invalid copyright is found at {fn}: expected={new_header}, actual={old_header}"
                )
            else:
                with open(str(fn), "w", encoding='utf-8') as fh:
                    fh.write(c.replace_file_header(old_header, new_header))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir', type=str, default='./')
    parser.add_argument('--check-only', action="store_true")
    args = parser.parse_args()
    main(args)
