# Password in a Haystack

Just about every machine in the [Top500 list of supercomputers](https://www.top500.org/lists/top500/list/2020/06/) runs a Linux (or at least Unix-like) operating system. Familiarity with the operating system's built-in utilities, along with common shell commands are a must.

Among the most helpful commands to know well is `grep` (which stands for "Global Regular Expression Print"). It's a way to search based on some user-provided pattern. At its simplest, `grep` works like cmd+F/ctrl+F. However, where it really shines is the ability to match patterns of characters through regular expressions, not just characters themselves. (Read more about Regular Expressions [here](https://en.wikipedia.org/wiki/Regular_expression).)

This directory has 2 password-protected PDFs, and the challenge is to use `grep` to help open them.

The three files

- `haystack_00.txt`
- `haystack_01.txt`
- `haystack_02.txt`

contain the hidden passwords, but it's up to you to find them! 

The passwords can be found by matching the string "password" or a representation of it using non-alphabetic characters. Think about how people may use 0s in place of Os, or @ instead of A. For example, consider the following lines of text from a file `demo.txt`:

```
Qd2ciwv2UqRHazew1cIg
nhJql8H68T6xngS2cdPL
FEHCJJKsi4v1gVC3A5mg
Z3e3kWrmJy3spP8K4Y3I
Dnvu2oWelNxhVBZQtn3s
PasswordIsHelloWorld
VyifvCmESfNGkYaXZtJe
KXG7E3KQrIW4UT42TqCF
```

In this sample, the string 'HelloWorld' would be a valid password. From the command line, we can quickly "grep through the file" to identify this.

```
$ grep "Password" ./demo.txt
PasswordIsHelloWorld
```

Take a look through the `grep` manual page by running `man grep`. You'll find that there are many options to help match patterns and format output to be more helpful. Additional flags can display the filename and line number where a match is found. (`q` will exit the man page) 

```
$ grep -Hin "password" ./demo.txt
./demo.txt:6:PasswordIsHelloWorld
```

> So to clarify, your task is to find the two passwords within the three `.txt` files. To do so, you will search for some form of the string "password", but it might include (for example) @ in place of a, 0 in place of O, some capital letters and some lowercase, etc. - so you will need to use a combination of `grep`'s options to search for all possible spellings of the string "password". The actual password that you are looking for will be given on the same line as the "password" string you find - similar to the example `demo.txt` above.

Once you have identified the passwords, go ahead and use them to open the two pdf documents in this directory. First, you must transfer the files to your personal computer. This can be accomplished by downloading them directly from git, using the `scp` command, or transfering them with Globus. If you are familiar with Globus you can use the NCCS Open DTN endpoint to access your home directory on Ascent. For `scp`, try the example command on a terminal on your own computer to transfer a file from Ascent to the current directory. (Read more about transferring data [here](https://docs.olcf.ornl.gov/data/transferring.html).) 

```
$ scp username@opendtn.ccs.ornl.gov:/ccsopen/home/username/path/to/file.ext .
```

#### A word on relevancy to HPC...

You may be thinking *"What does this have to do with supercomputing?"*, and this is an understandable question. After all, you could do this challenge on your laptop! The purpose of this challenge is more about fluency in a Unix-like environment than it is about running computationally complex simulations. To be most productive in an HPC environment, it's invaluable to be comfortable with the fundamental tools, including `grep`. 

Research teams running on the world's largest and most powerful computers like Summit often work with data at the scale of many terabytes, or even petabytes (much more data than the three text files included here), and pre-/post-processing or filtering of information on this scale generally requires more than a simple regular expression match.
