with open('E:\\Project\\HelloAgents\\hello_agents\\cli.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the indentation issue in cmd_init function
old = '""", encoding="utf-8")\n    \nprint(f"[OK] 项目已创建: {project_dir}")'
new = '""", encoding="utf-8")\n    \n    print(f"[OK] 项目已创建: {project_dir}")'

content = content.replace(old, new)

with open('E:\\Project\\HelloAgents\\hello_agents\\cli.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed')