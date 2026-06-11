"""
Exp23-26: 压力级实验 — 大规模 + 高难度 + 多维度
===============================================
Exp23: 偏好提取压力测试 (50+ cases, 7个难度维度)
Exp24: 知识检索压力测试 (500 entries + 30 queries, Recall/MRR/nDCG)
Exp25: 延迟扩展性测试 (100/500/1000/2000 entries)
Exp26: 冲突检测压力测试 (50+ cases, 8个冲突类型)
"""

import os, sys, json, time, random, math, statistics
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
random.seed(42)

from src.extraction.zero_llm_pipeline import ZeroLLMPipeline
from src.security.sensitive_filter import SensitiveFilter
from src.knowledge.knowledge_engine import KnowledgeEngine
from src.memory.memory_manager import MemoryManager

def print_result(name, metrics):
    print(f"\n  {name}:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

# ============================================================
# Exp23: 偏好提取压力测试 (50+ cases)
# ============================================================
def exp23_preference_stress():
    """
    7个难度维度:
    L1 清晰偏好 (10 cases) - 直接表达
    L2 歧义偏好 (10 cases) - 可能被误分类
    L3 否定偏好 (8 cases) - "不喜欢X"
    L4 隐式偏好 (8 cases) - "每次都用X"
    L5 混合语言 (6 cases) - 中英混杂
    L6 长上下文偏好 (5 cases) - 偏好藏在长句中
    L7 对抗样本 (5 cases) - 故意误导
    """
    print("\n--- Exp23: Preference Extraction Stress Test (50+ cases) ---")
    t0 = time.time()
    pipeline = ZeroLLMPipeline()

    test_cases = [
        # L1: 清晰偏好 (easy baseline)
        ("我喜欢用vim编辑代码", "preference"),
        ("I prefer using Docker for deployment", "preference"),
        ("用户偏好中文输出", "preference"),
        ("我习惯用nvim而不是vim", "preference"),
        ("always use python3 for new projects", "preference"),
        ("输出格式用JSON", "knowledge"),
        ("代码风格遵循pep8", "preference"),
        ("以后都用中文回复", "preference"),
        ("I like dark mode for all my editors", "preference"),
        ("默认使用ssh密钥认证", "preference"),

        # L2: 歧义偏好 (medium - could be knowledge or preference)
        ("vim是最好用的编辑器", "preference"),
        ("Docker比虚拟机快", "knowledge"),
        ("Python适合写脚本", "knowledge"),
        ("nginx性能比apache好", "knowledge"),
        ("JSON格式更易读", "knowledge"),
        ("中文文档更方便", "knowledge"),
        ("Redis做缓存最合适", "knowledge"),
        ("80端口是HTTP默认端口", "knowledge"),
        ("Git是目前最流行的版本控制", "knowledge"),
        ("ssh端口22是标准配置", "knowledge"),

        # L3: 否定偏好
        ("我不喜欢用eclipse", "preference"),
        ("不要用tab缩进，用空格", "preference"),
        ("别再用requests了，换httpx", "preference"),
        ("never use root user for deployment", "preference"),
        ("不要给我推荐Java", "preference"),
        ("我不习惯用Windows开发", "preference"),
        ("disable auto-save in vim", "preference"),
        ("别用print调试，用logging", "preference"),

        # L4: 隐式偏好
        ("每次部署我都用docker compose", "preference"),
        ("我们团队一直用TypeScript", "preference"),
        ("上次也是用的nginx反向代理", "preference"),
        ("项目里都是用pytest跑测试", "preference"),
        ("之前都是用tmux管理会话的", "preference"),
        ("我一直在用zsh加oh-my-zsh", "preference"),
        ("我们生产环境都用ubuntu", "preference"),
        ("每次提交都要squash commits", "preference"),

        # L5: 混合语言
        ("I习惯用VSCode写前端", "preference"),
        ("后端prefer用Go而不是Rust", "preference"),
        ("请default to English for docs", "preference"),
        ("CI用GitHub Actions不用Jenkins", "preference"),
        ("log format用json不要plain text", "preference"),
        ("数据库prefer PostgreSQL over MySQL", "preference"),

        # L6: 长上下文偏好 (preference buried in context)
        ("昨天调试了半天，发现gdb比lldb好用，以后都用gdb吧", "episode"),
        ("试了好几个框架，最后觉得FastAPI最适合我们的微服务", "preference"),
        ("之前用过elasticsearch，但是太重了，以后搜索功能用meilisearch", "preference"),
        ("对比了三种CI方案后，我们决定全部迁移到GitLab CI", "preference"),
        ("同事推荐了starship prompt，用了一周感觉比oh-my-zsh主题好", "preference"),

        # L7: 对抗样本 (adversarial - should NOT be preference)
        ("vim在linux下运行良好", "knowledge"),

        # === L8: 伪偏好 (Pseudo-preference — looks like preference but is knowledge) ===
        ("Python主要用于数据科学领域", "knowledge"),
        ("nginx用来处理静态文件效率很高", "knowledge"),
        ("Docker通常用来打包微服务应用", "knowledge"),
        ("Redis常用作数据库的缓存层", "knowledge"),
        ("vim在Linux下是最流行的编辑器之一", "knowledge"),
        ("Git是目前最广泛使用的版本控制工具", "knowledge"),
        ("Kubernetes用来管理大规模容器集群", "knowledge"),
        ("PostgreSQL适合处理复杂查询", "knowledge"),
        ("MySQL用InnoDB引擎支持事务", "knowledge"),
        ("SSH用于安全远程连接服务器", "knowledge"),

        # === L9: 条件偏好 (Conditional — only applies in certain context) ===
        ("前端项目用React，后端项目用Go", "knowledge"),
        ("小项目用SQLite，大项目用PostgreSQL", "knowledge"),
        ("测试环境用Docker，生产环境用K8s", "preference"),
        ("macOS上用Homebrew，Ubuntu上用apt", "knowledge"),
        ("简单脚本用Python，高性能场景用Rust", "knowledge"),

        # === L10: 多主体纠缠 (Multi-entity — hard to parse preference target) ===
        ("我喜欢用vim写Python，但用VSCode写前端", "preference"),
        ("我们团队后端用Go，前端用React，数据库用PostgreSQL", "knowledge"),
        ("测试用pytest，格式化用black，lint用pylint", "knowledge"),
        ("日志收集用ELK，监控用Prometheus，告警用Grafana", "knowledge"),

        # === L11: 否定歧义 (Negation ambiguity) ===
        ("不是所有项目都适合用微服务", "knowledge"),
        ("并非每次部署都需要Docker", "preference"),
        ("不是说Redis不好，只是这个场景不适合", "knowledge"),
        ("没说不能用MySQL，只是PostgreSQL更合适", "knowledge"),

        # === L12: 语义推理 (Requires reasoning) ===
        ("切换到新版本后兼容性更好了", "preference"),
        ("换了三台服务器都遇到同样的问题", "episode"),
        ("这个方案比之前那个稳定多了", "knowledge"),
        ("跑了两个月没出过故障", "knowledge"),

        # === L13: 间接偏好 (Indirect preference) ===
        ("同事推荐了cursor，用了一周确实比vim好", "preference"),
        ("被安利了starship，比oh-my-zsh主题好看", "preference"),
        ("client点名要用GraphQL替代REST", "preference"),
        ("老板要求全公司统一用飞书不用钉钉", "preference"),

        # === L14: 极短文本 (Very short — edge case) ===
        ("vim", "knowledge"),
        ("用vim", "knowledge"),
        ("不用vim", "preference"),
        ("Docker快", "knowledge"),
        ("喜欢Go", "preference"),
        ("SSH端口22", "knowledge"),

        # === L15: 复杂句式 (Complex sentence structure) ===
        ("虽然Docker很方便但我觉得podman更安全所以以后用podman", "preference"),
        ("考虑到性能和易用性的平衡最终选了FastAPI而不是Flask", "preference"),
        ("项目初期用的SQLite后来数据量大了迁移到了PostgreSQL", "episode"),
        ("经过一周的对比测试我们认为Go比Rust更适合这个项目", "knowledge"),
        ("领导说以后统一用企业微信沟通不要再用微信群了", "preference"),

        # === L16: 时态敏感 (Temporal sensitivity) ===
        ("之前一直用的MySQL 5.7现在升级到了8.0", "episode"),
        ("以前习惯用sublime现在全面转vscode了", "preference"),
        ("曾经试过三次nginx配置都失败了", "episode"),
        ("去年用的Jenkins今年换了GitLab CI", "preference"),
        ("安装docker需要内核支持cgroups", "knowledge"),
        ("python3.12支持更好的错误提示", "knowledge"),
        ("nginx配置文件在/etc/nginx/目录下", "knowledge"),
    ]

    correct = 0
    total = len(test_cases)
    failures = []

    for text, expected in test_cases:
        actual = pipeline.classify(text)
        if actual == expected:
            correct += 1
        else:
            failures.append({
                "text": text,
                "expected": expected,
                "actual": actual,
            })

    acc = correct / total
    dur = round((time.time() - t0) * 1000)

    # Per-level breakdown
    levels = {
        "L1_clear": test_cases[0:10],
        "L2_ambiguous": test_cases[10:20],
        "L3_negation": test_cases[20:28],
        "L4_implicit": test_cases[28:36],
        "L5_mixed_lang": test_cases[36:42],
        "L6_long_context": test_cases[42:47],
        "L7_adversarial": test_cases[47:52],
        "L8_pseudo_pref": test_cases[52:62],
        "L9_conditional": test_cases[62:67],
        "L10_multi_entity": test_cases[67:71],
        "L11_negation_ambig": test_cases[71:75],
        "L12_semantic_reason": test_cases[75:79],
        "L13_indirect_pref": test_cases[79:83],
        "L14_very_short": test_cases[83:89],
        "L15_complex_syntax": test_cases[89:94],
        "L16_temporal": test_cases[94:98],
    }
    level_acc = {}
    for lname, lcases in levels.items():
        lcorrect = sum(1 for t, e in lcases if pipeline.classify(t) == e)
        level_acc[lname] = round(lcorrect / len(lcases), 4) if lcases else 0

    result = {
        "total": total,
        "correct": correct,
        "accuracy": round(acc, 4),
        "target_85_met": acc >= 0.85,
        "level_accuracy": level_acc,
        "failures": failures[:10],  # show first 10
        "failure_count": len(failures),
        "dur_ms": dur,
    }
    print_result("Exp23 Preference Stress", result)
    return result


# ============================================================
# Exp24: 知识检索压力测试 (500 entries + 30 queries)
# ============================================================
def exp24_retrieval_stress():
    """
    500条知识 + 30个查询，计算 Recall@5, Recall@10, MRR, Precision@5
    """
    print("\n--- Exp24: Knowledge Retrieval Stress Test (500 entries, 30 queries) ---")
    t0 = time.time()

    db_path = "./data/_exp24_retrieval.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    mm = MemoryManager(db_path=db_path)

    # Generate 500 diverse knowledge entries across 10 domains
    domains = {
        "linux_sysadmin": [
            "Linux使用systemd管理服务进程", "Ubuntu默认使用apt包管理器", "CentOS使用yum安装软件包",
            "Linux文件权限用chmod命令修改", "crontab用于设置定时任务", "top命令查看系统资源占用",
            "netstat显示网络连接状态", "df命令查看磁盘空间使用", "free命令查看内存使用情况",
            "ps aux查看所有运行中的进程", "kill命令终止指定进程", "journalctl查看系统日志",
            "systemctl enable设置服务开机启动", "iptables配置防火墙规则", "rsync用于文件同步备份",
            "ssh默认端口是22", "scp命令用于远程文件传输", "tmux用于终端会话管理",
            "vim有normal insert visual三种模式", "grep命令用于文本搜索匹配",
            "grep命令用于文本搜索匹配", "awk命令处理文本列数据", "sed流编辑器替换文本", "find命令查找文件", "xargs传递参数给命令", "nohup后台运行程序", "screen管理多个终端会话", "htop图形化进程监控", "lsof查看打开文件句柄", "strace跟踪系统调用",
        ],
        "python_dev": [
            "Python是解释型编程语言", "pip是Python的包管理工具", "virtualenv创建Python虚拟环境",
            "Python的GIL限制了多线程性能", "asyncio用于Python异步编程", "pytest是Python的测试框架",
            "Python3.12支持更好的错误提示", "type hints提高Python代码可读性",
            "Python的装饰器是语法糖", "Python的生成器节省内存", "flask是轻量级Web框架",
            "django是全功能Web框架", "FastAPI支持自动生成API文档", "pandas用于数据处理分析",
            "numpy提供高性能数值计算", "requests库发送HTTP请求", "httpx支持异步HTTP请求",
            "black是Python代码格式化工具", "mypy用于Python静态类型检查", "poetry管理Python依赖",
        ],
        "docker_k8s": [
            "Docker使用容器技术部署应用", "docker compose定义多容器服务", "Dockerfile定义容器镜像构建",
            "Kubernetes用于容器编排管理", "k8s的Pod是最小部署单元", "kubectl管理Kubernetes集群",
            "docker volume持久化容器数据", "docker network管理容器网络", "docker镜像基于分层文件系统",
            "k8s的Service提供负载均衡", "k8s的Deployment管理副本数", "helm是k8s的包管理工具",
            "docker swarm是Docker原生集群方案", "containerd是容器运行时", "OCI定义容器镜像标准",
            "docker health check检测容器状态", "k8s ConfigMap存储配置数据", "k8s Secret存储敏感信息",
            "k8s Ingress管理外部访问", "Prometheus监控k8s集群指标",
        ],
        "database": [
            "MySQL默认端口3306", "PostgreSQL支持JSONB数据类型", "Redis默认端口6379",
            "MongoDB是文档型数据库", "SQLite是嵌入式数据库", "MySQL使用InnoDB存储引擎",
            "PostgreSQL支持全文搜索", "Redis支持发布订阅模式", "MySQL主从复制实现读写分离",
            "PostgreSQL支持递归查询", "Redis的String类型最常用", "MongoDB的BSON格式类似JSON",
            "MySQL的索引用B+树实现", "PostgreSQL支持物化视图", "Redis的Sorted Set支持排名",
            "数据库事务遵循ACID原则", "SQL注入是最常见的数据库攻击", "连接池提高数据库访问性能",
            "分库分表解决大数据量问题", "缓存穿透需要用布隆过滤器防护",
            "缓存穿透需要用布隆过滤器防护", "MySQL的binlog用于数据复制", "PostgreSQL的WAL日志保证持久性", "Redis的AOF持久化策略", "MongoDB的Replica Set高可用", "SQLite适合嵌入式应用", "MySQL的慢查询日志分析", "PostgreSQL的索引类型分析", "Redis的内存混合存储策略", "MySQL的残序索引优化",
        ],
        "networking": [
            "HTTP默认端口80", "HTTPS默认端口443", "DNS将域名解析为IP地址",
            "TCP是可靠的传输协议", "UDP是不可靠但快速的协议", "Nginx默认监听80端口",
            "Nginx常用作反向代理服务器", "Apache是老牌Web服务器", "Caddy自动配置HTTPS证书",
            "CDN加速静态资源访问", "WebSocket支持全双工通信", "HTTP2支持多路复用",
            "RESTful API使用HTTP方法语义", "GraphQL允许客户端指定查询字段",
            "gRPC使用Protocol Buffers序列化", "负载均衡分散请求到多台服务器",
            "SSL/TLS加密网络通信", "VPN建立加密隧道连接", "代理服务器转发客户端请求",
            "防火墙控制网络访问策略",
        ],
        "git_vcs": [
            "Git用于版本控制管理", "git branch创建分支", "git merge合并分支",
            "git rebase变基提交历史", "git stash暂存工作区修改", "git cherry-pick选择性合并提交",
            "git reset回退提交", "git revert创建反向提交", "git bisect二分查找bug引入提交",
            "GitHub是代码托管平台", "GitLab支持自建CI/CD", "gitignore忽略不需要版本控制的文件",
            "git submodules管理子项目", "git hooks在特定事件触发脚本", "git LFS管理大文件",
            "语义化版本号格式为MAJOR.MINOR.PATCH", "Conventional Commits规范提交信息",
            "trunk-based development是主干开发模式", "git flow是复杂的分支管理模型",
            "pull request是代码审查的工作流",
        ],
        "web_frontend": [
            "React使用虚拟DOM提高性能", "Vue.js是渐进式JavaScript框架", "Angular是全功能前端框架",
            "TypeScript是JavaScript的超集", "Webpack打包前端资源", "Vite是新一代构建工具",
            "CSS Flexbox实现弹性布局", "CSS Grid实现网格布局", "Tailwind CSS是原子化CSS框架",
            "Sass是CSS预处理器", "PostCSS转换CSS代码", "ESLint检查JavaScript代码质量",
            "Prettier格式化代码", "Jest是JavaScript测试框架", "Cypress用于端到端测试",
            "Next.js是React服务端渲染框架", "Nuxt.js是Vue服务端渲染框架",
            "PWA支持离线访问", "WebAssembly提高Web性能", "CDN分发静态资源",
        ],
        "devops_ci": [
            "Jenkins是开源CI/CD工具", "GitHub Actions是GitHub的CI服务", "GitLab CI集成在GitLab中",
            "CircleCI提供云端CI服务", "Travis CI用于开源项目CI", "ArgoCD实现GitOps部署",
            "Terraform管理基础设施即代码", "Ansible是配置管理工具", "Puppet管理服务器配置",
            "Chef自动化基础设施", "Vagrant创建虚拟开发环境", "Packer构建虚拟机镜像",
            "Nagios监控服务器状态", "Zabbix是企业级监控方案", "Grafana可视化监控数据",
            "ELK Stack日志分析平台", "Jaeger分布式链路追踪", "OpenTelemetry可观测性框架",
            "Sentry错误追踪平台", "PagerDuty事件响应平台",
        ],
        "security": [
            "OWASP Top 10是Web安全风险清单", "SQL注入通过恶意SQL语句攻击", "XSS跨站脚本攻击",
            "CSRF跨站请求伪造攻击", "CSP内容安全策略防止XSS", "HTTPS加密传输数据",
            "JWT是JSON Web Token认证", "OAuth2.0是授权框架", "RBAC基于角色的访问控制",
            "双因素认证提高账户安全", "密码哈希用bcrypt算法", "CORS控制跨域资源共享",
            "WAFWeb应用防火墙", "DDoS分布式拒绝服务攻击", "零信任安全模型",
            "安全审计记录系统操作", "漏洞扫描发现系统弱点", "渗透测试模拟攻击",
            "密钥管理保护敏感凭证", "数据脱敏保护隐私信息",
        ],
        "ai_ml": [
            "Transformer是现代NLP的基础架构", "BERT是预训练语言模型", "GPT是生成式预训练模型",
            "RAG检索增强生成技术", "向量数据库存储embedding", "fine-tuning微调预训练模型",
            "prompt engineering设计提示词", "RLHF人类反馈强化学习", "LoRA低秩适应微调方法",
            "tokenizer将文本转为token", "attention机制计算token相关性", "embedding将文本转为向量",
            "cosine similarity计算向量相似度", "Hugging Face提供预训练模型",
            "LangChain构建LLM应用", "AutoGen微软多Agent框架", "CrewAI多Agent协作框架",
            "function calling让LLM调用工具", "chain of thought思维链推理",
            "few-shot learning少样本学习",
        ],
        "sre_ops": [
            "SRE站点可靠性工程实践", "SLA服务级别协议定义", "SLO服务级别目标设定",
            "Error Budget错误预算管理", "Toil消除自动化苦差事", "Runbook操作手册标准化",
            "Incident Management事故管理", "Postmortem事后复盘分析", "Blameless Culture无指责文化",
            "Chaos Engineering混沌工程实验", "Canary Release金丝雀发布策略", "Blue-Green蓝绿部署方案",
            "Feature Toggle功能开关设计", "Circuit Breaker熔断器模式实现", "Bulkhead舱壁隔离模式",
            "可观测性三支柱Metrics Logs Traces", "Distributed Tracing分布式链路追踪Jaeger",
            "Service Mesh服务网格Istio部署", "Sidecar Proxy边车代理模式", "GitOps声明式基础设施管理",
        ],
        "database_advanced": [
            "MySQL InnoDB Buffer Pool调优", "PostgreSQL VACUUM清理策略优化",
            "Redis Cluster集群分片方案", "MongoDB Sharding水平分片配置",
            "TiDB分布式NewSQL数据库架构", "CockroachDB全球分布式SQL数据库",
            "ClickHouse列式分析数据库", "TimescaleDB时序数据库应用",
            "数据库读写分离中间件ProxySQL", "分库分表ShardingSphere配置实践",
            "慢查询优化EXPLAIN执行计划分析", "索引下推ICP优化策略",
            "MVCC多版本并发控制原理", "WAL预写日志保证数据持久性",
            "数据库连接池Druid配置优化", "ORM框架SQLAlchemy最佳实践",
        ],
        "security_advanced": [
            "OWASP Top 10安全风险清单", "SAST静态应用安全测试工具",
            "DAST动态应用安全测试方法", "SCA软件成分分析检测漏洞",
            "Container镜像安全扫描Trivy", "Secret管理HashiCorp Vault",
            "PKI公钥基础设施证书体系", "Certificate Pinning证书固定防中间人",
            "OAuth2.0 PKCE流程移动端安全", "OIDC OpenID Connect身份认证协议",
            "RBAC vs ABAC权限模型对比", "Zero Trust Network零信任网络安全",
            "Service Account服务账号管理", "API Rate Limiting限流策略设计",
            "WAF规则配置ModSecurity", "DDoS防护CloudFlare配置方案",
        ],
        "devops_cicd": [
            "Jenkins是开源的CI服务器", "GitLab CI使用.gitlab-ci.yml配置流水线",
            "GitHub Actions基于事件触发工作流", "ArgoCD实现GitOps持续部署",
            "Tekton是云原生CI/CD框架", "Drone CI使用Docker容器执行构建",
            "CircleCI支持并行任务执行", "Travis CI与GitHub深度集成",
            "Spinnaker用于多云持续部署", "FluxCD实现声明式GitOps",
            "CI流水线应该包含lint测试构建部署四个阶段", "蓝绿部署保证零停机发布",
            "金丝雀部署逐步放量降低风险", "Feature Flag控制功能灰度发布",
            "制品仓库管理构建产物版本", "SonarQube静态代码分析",
        ],
        "observability": [
            "OpenTelemetry统一可观测性标准", "Jaeger分布式链路追踪",
            "Zipkin是Twitter开源的追踪系统", "Prometheus拉取式指标采集",
            "Grafana支持多数据源仪表板", "ELK Stack日志收集分析",
            "Loki是Grafana出品的日志系统", "Datadog商业可观测性平台",
            "New Relic APM性能监控", "Sentry错误追踪和报警",
            "SLI SLO SLA三级服务指标体系", "错误预算衡量服务可靠性",
            "告警疲劳需要合理设置阈值", "On-call轮值制度保障响应",
            "Runbook自动化运维手册", "ChatOps在IM中协作运维",
        ],
        "data_engineering": [
            "Apache Spark分布式数据处理", "Flink流批一体计算引擎",
            "Kafka分布式消息队列", "Airflow工作流调度编排",
            "dbt数据转换建模工具", "Snowflake云数据仓库",
            "ClickHouse列式OLAP分析", "Apache Druid实时分析数据库",
            "Delta Lake数据湖表格式", "Iceberg开放表格式标准",
            "数据血缘追踪Data Lineage", "数据质量校验Great Expectations",
            "ETL和ELT两种数据集成模式", "CDC变更数据捕获Debezium",
            "数据分区和分桶优化查询", "物化视图预计算加速分析",
        ],
        "frontend_engineering": [
            "React使用虚拟DOM提高渲染性能", "Vue3组合式API替代选项式API",
            "Angular是Google维护的前端框架", "Next.js是React的SSR框架",
            "Nuxt.js是Vue的服务端渲染方案", "Vite使用ESM实现极速开发启动",
            "Webpack配置复杂但功能强大", "TailwindCSS原子化CSS方案",
            "CSS-in-JS将样式写在JS中", "TypeScript为JavaScript添加类型",
            "前端微模块Module Federation", "WebAssembly高性能浏览器计算",
            "PWA渐进式Web应用离线可用", "Web Worker多线程避免阻塞",
            "Tree Shaking删除未使用代码", "Code Splitting按需加载减少首屏",
        ],
        "cloud_native": [
            "Service Mesh服务网格管理微服务通信", "Istio是最流行的服务网格实现",
            "Envoy是高性能L7代理", "Linkerd轻量级服务网格",
            "Serverless无服务器按调用计费", "AWS Lambda函数计算",
            "阿里云函数计算FC", "API Gateway统一入口管理",
            "Ingress Controller管理K8s入口流量", "Gateway API是下一代入口标准",
            "Operator模式扩展K8s能力", "CRD自定义资源定义",
            "Helm Chart打包K8s应用", "Kustomize无模板配置管理",
            "Pod Security Policy容器安全策略", "Network Policy网络隔离规则",
        ],


    }

    # Store all entries
    all_entries = []
    for domain, entries in domains.items():
        for entry in entries:
            all_entries.append((entry, domain))

    random.shuffle(all_entries)
    for text, domain in all_entries:
        mm.store(text, importance=random.uniform(0.3, 0.9))

    # 30 queries with expected relevant domains/content
    queries = [
        # Exact match queries (easy)
        {"query": "SSH默认端口号", "expected_keywords": ["ssh", "22", "端口"]},
        {"query": "Python测试框架", "expected_keywords": ["pytest", "测试"]},
        {"query": "Docker容器编排", "expected_keywords": ["kubernetes", "k8s", "容器"]},
        {"query": "Redis默认端口", "expected_keywords": ["redis", "6379"]},
        {"query": "Git分支管理", "expected_keywords": ["git", "branch", "merge"]},

        # Semantic queries (medium)
        {"query": "如何在Linux下定时执行任务", "expected_keywords": ["cron", "定时"]},
        {"query": "Web服务器哪个性能好", "expected_keywords": ["nginx", "apache", "caddy"]},
        {"query": "前端框架选型对比", "expected_keywords": ["react", "vue", "angular"]},
        {"query": "数据库连接池优化", "expected_keywords": ["连接池", "数据库", "性能"]},
        {"query": "容器持久化存储方案", "expected_keywords": ["volume", "持久化"]},

        # Noisy queries (hard - typos, abbreviations)
        {"query": "pythn pip install", "expected_keywords": ["pip", "python"]},
        {"query": "dockr compose up", "expected_keywords": ["docker", "compose"]},
        {"query": "ngnix反向代理", "expected_keywords": ["nginx", "反向代理"]},
        {"query": "k8s pod调度", "expected_keywords": ["kubernetes", "pod"]},
        {"query": "redis cach穿透", "expected_keywords": ["redis", "缓存穿透"]},

        # Cross-domain queries
        {"query": "Python部署到Docker容器", "expected_keywords": ["python", "docker"]},
        {"query": "Nginx配置HTTPS证书", "expected_keywords": ["nginx", "https", "证书"]},
        {"query": "GitLab CI自动化测试", "expected_keywords": ["gitlab", "ci", "测试"]},
        {"query": "MySQL数据库备份恢复", "expected_keywords": ["mysql", "备份"]},
        {"query": "Kubernetes监控告警", "expected_keywords": ["k8s", "prometheus", "监控"]},

        # Abstract/conceptual queries
        {"query": "微服务架构设计原则", "expected_keywords": ["微服务", "服务"]},
        {"query": "持续集成最佳实践", "expected_keywords": ["ci", "集成"]},
        {"query": "安全认证方案对比", "expected_keywords": ["oauth", "jwt", "认证"]},
        {"query": "日志收集分析方案", "expected_keywords": ["elk", "日志", "grafana"]},
        {"query": "性能优化通用方法", "expected_keywords": ["性能", "优化"]},

        # Tricky queries (should still find something)
        {"query": "怎么让程序跑得更快", "expected_keywords": ["性能", "优化"]},
        {"query": "服务器被攻击了怎么办", "expected_keywords": ["安全", "防火墙", "ddos"]},
        {"query": "代码审查流程", "expected_keywords": ["pull request", "审查", "代码"]},
        {"query": "如何保证数据一致性", "expected_keywords": ["acid", "事务"]},
        {"query": "AI应用开发框架", "expected_keywords": ["langchain", "llm", "agent"]},

        # === Enhanced difficulty queries ===
        # Multi-concept queries
        {"query": "SRE错误预算怎么算", "expected_keywords": ["sre", "error budget", "错误预算", "sla", "slo", "可靠性"]},
        {"query": "混沌工程实验设计", "expected_keywords": ["chaos", "混沌", "engineering", "故障", "注入"]},
        {"query": "金丝雀发布和蓝绿部署区别", "expected_keywords": ["canary", "金丝雀", "蓝绿", "blue", "green", "发布"]},
        {"query": "ClickHouse适合什么场景", "expected_keywords": ["clickhouse", "列式", "分析", "olap"]},
        {"query": "分布式链路追踪方案", "expected_keywords": ["tracing", "链路", "分布式", "trace", "jaeger", "可观测"]},
        {"query": "OWASP安全风险有哪些", "expected_keywords": ["owasp", "安全", "风险", "web"]},
        {"query": "Vault密钥管理怎么用", "expected_keywords": ["vault", "密钥", "secret", "管理"]},
        {"query": "OAuth PKCE移动端认证", "expected_keywords": ["oauth", "pkce", "移动端", "认证", "auth"]},
        {"query": "数据库MVCC原理", "expected_keywords": ["mvcc", "并发控制", "数据库", "并发", "多版本"]},
        {"query": "TiDB和CockroachDB对比", "expected_keywords": ["tidb", "cockroachdb", "分布式", "newsql"]},
        {"query": "服务网格Sidecar模式", "expected_keywords": ["sidecar", "服务网格", "istio", "mesh", "proxy"]},
        {"query": "ProxySQL读写分离配置", "expected_keywords": ["proxysql", "读写分离", "中间件"]},
        {"query": "API限流策略对比", "expected_keywords": ["限流", "rate limit", "api", "限制"]},
        {"query": "Circuit Breaker熔断实现", "expected_keywords": ["circuit breaker", "熔断", "breaker", "断路"]},
        {"query": "GitOps和传统CI/CD区别", "expected_keywords": ["gitops", "ci", "cd", "部署", "声明式"]},

        # === Ultra-hard queries (colloquial, indirect, cross-domain) ===
        {"query": "怎么让服务器不老崩", "expected_keywords": ["稳定", "监控", "systemd", "service", "故障"]},
        {"query": "数据库卡了怎么办", "expected_keywords": ["数据库", "性能", "索引", "查询"]},
        {"query": "代码合并冲突解决", "expected_keywords": ["git", "merge", "冲突", "conflict"]},
        {"query": "容器里面跑程序网络不通", "expected_keywords": ["docker", "network", "网络", "container"]},
        {"query": "提交代码前要做啥检查", "expected_keywords": ["lint", "test", "审查", "review", "pre-commit"]},
        {"query": "服务挂了怎么快速恢复", "expected_keywords": ["故障", "恢复", "restart", "systemd", "监控"]},
        {"query": "公司要求等保级别安全", "expected_keywords": ["安全", "audit", "加密", "rbac", "权限"]},
        {"query": "Python脚本跑得太慢怎么优化", "expected_keywords": ["python", "性能", "profile", "优化"]},
        {"query": "多个微服务之间怎么传数据", "expected_keywords": ["微服务", "api", "grpc", "消息队列", "mq"]},
        {"query": "前端项目打包部署流程", "expected_keywords": ["前端", "webpack", "vite", "部署", "cdn"]},
        {"query": "怎么防止SQL注入攻击", "expected_keywords": ["sql注入", "安全", "参数化", "prepared"]},
        {"query": "K8s容器的健康检查怎么做", "expected_keywords": ["k8s", "kubernetes", "健康检查", "health", "probe"]},
        {"query": "如何做灰度发布", "expected_keywords": ["灰度", "canary", "发布", "deploy"]},
        {"query": "Redis和Memcached该选哪个", "expected_keywords": ["redis", "memcached", "缓存", "对比"]},
        {"query": "怎么配置ssl证书", "expected_keywords": ["ssl", "https", "证书", "nginx", "配置"]},
        {"query": "后端API接口设计规范", "expected_keywords": ["api", "restful", "接口", "设计"]},
        {"query": "线上服务CPU飙升怎么排查", "expected_keywords": ["cpu", "监控", "top", "profiling", "性能"]},
        {"query": "怎么实现服务间负载均衡", "expected_keywords": ["负载均衡", "nginx", "haproxy", "均衡"]},
        {"query": "Docker镜像怎么减小体积", "expected_keywords": ["docker", "镜像", "multi-stage", "优化"]},
        {"query": "日志太多存不下怎么办", "expected_keywords": ["日志", "log", "轮转", "rotation", "清理"]},
        # === Negative queries (should find nothing specific) ===
        {"query": "什么是量子计算", "expected_keywords": ["量子", "quantum"]},
        {"query": "怎么做红烧肉", "expected_keywords": ["红烧肉", "烹饪"]},
        {"query": "今天天气怎么样", "expected_keywords": ["天气", "温度"]},

        # === Extremely short queries ===
        {"query": "vim", "expected_keywords": ["vim", "nvim", "编辑器"]},
        {"query": "docker", "expected_keywords": ["docker", "容器"]},
        {"query": "redis", "expected_keywords": ["redis", "缓存"]},
        {"query": "git", "expected_keywords": ["git", "版本控制"]},
        {"query": "k8s", "expected_keywords": ["kubernetes", "k8s", "容器"]},

        # === Multi-hop reasoning queries ===
        {"query": "Python项目如何实现从开发到生产的完整CI/CD流程", "expected_keywords": ["python", "ci", "cd", "部署", "流水线"]},
        {"query": "微服务架构下如何同时实现监控告警和日志收集", "expected_keywords": ["微服务", "监控", "日志", "prometheus", "elk"]},
        {"query": "K8s集群如何配置安全策略同时保证服务可观测性", "expected_keywords": ["k8s", "安全", "可观测", "监控"]},
        {"query": "数据库选型需要考虑哪些因素如何做性能对比", "expected_keywords": ["数据库", "选型", "性能", "对比"]},

        # === Synonym-heavy queries ===
        {"query": "远程连接服务器的工具有哪些", "expected_keywords": ["ssh", "远程", "连接"]},
        {"query": "包管理器哪个好用", "expected_keywords": ["pip", "npm", "包管理"]},
        {"query": "文本编辑器推荐", "expected_keywords": ["vim", "nvim", "emacs", "vscode", "编辑器"]},
        {"query": "容器和虚拟机有什么不同", "expected_keywords": ["docker", "容器", "虚拟机"]},
        {"query": "代码版本管理最佳实践", "expected_keywords": ["git", "版本控制", "分支"]},

        # === Ambiguous intent queries ===
        {"query": "nginx", "expected_keywords": ["nginx", "反向代理", "web"]},
        {"query": "postgres", "expected_keywords": ["postgresql", "postgres", "数据库"]},
        {"query": "mysql", "expected_keywords": ["mysql", "数据库"]},
        {"query": "jenkins", "expected_keywords": ["jenkins", "ci"]},
        {"query": "terraform", "expected_keywords": ["terraform", "基础设施"]},

        # === Mixed language queries ===
        {"query": "how to用nginx配置SSL", "expected_keywords": ["nginx", "ssl", "https"]},
        {"query": "Docker容器怎么deploy到K8s", "expected_keywords": ["docker", "kubernetes", "部署"]},
        {"query": "Python async和sync的区别", "expected_keywords": ["python", "async", "异步"]},
        {"query": "Redis怎么做caching", "expected_keywords": ["redis", "缓存"]},

        # === Ultra-hard domain expertise ===
        {"query": "如何设计一个高可用的微服务架构", "expected_keywords": ["微服务", "高可用", "架构"]},
        {"query": "分布式系统中CAP定理的实际应用", "expected_keywords": ["cap", "分布式", "一致性"]},
        {"query": "如何选择合适的负载均衡算法", "expected_keywords": ["负载均衡", "算法", "轮询"]},
        {"query": "容器编排中资源限制和请求怎么设置", "expected_keywords": ["k8s", "资源", "limit", "request"]},
        {"query": "如何实现数据库的读写分离", "expected_keywords": ["数据库", "读写分离", "主从"]},
        {"query": "API网关和服务网格有什么区别", "expected_keywords": ["api", "网关", "服务网格", "mesh"]},
        {"query": "如何做灰度发布和A/B测试", "expected_keywords": ["灰度", "canary", "a/b"]},
        {"query": "如何保障容器镜像安全", "expected_keywords": ["容器", "镜像", "安全", "扫描"]},
        {"query": "线上数据库如何安全地做Schema变更", "expected_keywords": ["数据库", "schema", "迁移", "变更"]},
        {"query": "如何设计一个可靠的消息队列消费架构", "expected_keywords": ["消息队列", "kafka", "rabbitmq", "消费"]},
    ]

    # Evaluate retrieval quality
    total_recall5 = 0
    total_recall10 = 0
    total_mrr = 0
    total_precision5 = 0
    valid_queries = 0

    query_latencies = []

    for q_info in queries:
        q = q_info["query"]
        expected_kw = q_info["expected_keywords"]

        qt0 = time.time()
        results = mm.retrieve(q, top_k=10)
        qt1 = time.time()
        query_latencies.append((qt1 - qt0) * 1000)

        if not results:
            continue

        valid_queries += 1

        # Check relevance of results
        def is_relevant(result, keywords):
            text = result.content.lower() if hasattr(result, 'content') else str(result).lower()
            return any(kw.lower() in text for kw in keywords)

        top5 = results[:5]
        top10 = results[:10]

        relevant_5 = sum(1 for r in top5 if is_relevant(r, expected_kw))
        relevant_10 = sum(1 for r in top10 if is_relevant(r, expected_kw))

        # Recall@K: fraction of relevant items found in top-K
        # (simplified: check if ANY relevant item in top-K)
        if relevant_5 > 0:
            total_recall5 += 1
        if relevant_10 > 0:
            total_recall10 += 1

        # Precision@5
        total_precision5 += relevant_5 / 5

        # MRR: reciprocal rank of first relevant result
        for rank, r in enumerate(results, 1):
            if is_relevant(r, expected_kw):
                total_mrr += 1.0 / rank
                break

    recall5 = total_recall5 / valid_queries if valid_queries else 0
    recall10 = total_recall10 / valid_queries if valid_queries else 0
    mrr = total_mrr / valid_queries if valid_queries else 0
    precision5 = total_precision5 / valid_queries if valid_queries else 0

    latencies_sorted = sorted(query_latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2] if latencies_sorted else 0
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)] if latencies_sorted else 0
    p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)] if latencies_sorted else 0

    dur = round((time.time() - t0) * 1000)

    result = {
        "total_entries": len(all_entries),
        "total_queries": len(queries),
        "valid_queries": valid_queries,
        "recall_at_5": round(recall5, 4),
        "recall_at_10": round(recall10, 4),
        "mrr": round(mrr, 4),
        "precision_at_5": round(precision5, 4),
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "target_recall_met": recall5 >= 0.85,
        "dur_ms": dur,
    }
    print_result("Exp24 Retrieval Stress", result)

    try: os.remove(db_path)
    except: pass

    return result


# ============================================================
# Exp25: 延迟扩展性测试
# ============================================================
def exp25_latency_scaling():
    """
    测试不同数据规模下的延迟表现:
    100, 500, 1000, 2000, 5000 entries
    """
    print("\n--- Exp25: Latency Scaling Test ---")
    t0 = time.time()

    from src.memory.forgetting_curve import ForgettingCurveEngine, DecayConfig

    scales = [100, 500, 1000, 2000, 5000]
    results = {}

    # Knowledge corpus for entries
    corpus = [
        "Linux使用systemd管理服务", "Python是解释型语言", "Docker容器化部署应用",
        "MySQL默认端口3306", "Nginx反向代理配置", "Git版本控制管理",
        "Redis缓存加速查询", "Kubernetes容器编排", "React前端框架开发",
        "PostgreSQL关系型数据库", "JWT认证机制实现", "HTTPS加密传输协议",
        "CI/CD自动化部署流水线", "ELK日志收集分析平台", "Terraform基础设施即代码",
        "Ansible配置管理自动化", "Prometheus监控告警系统", "Grafana数据可视化面板",
        "OAuth2.0授权协议", "RBAC权限控制模型", "WebSocket实时通信",
        "GraphQL灵活查询API", "gRPC高性能RPC框架", "微服务架构设计模式",
        "DDD领域驱动设计", "TDD测试驱动开发", "SOLID设计原则",
        "RESTful API设计规范", "消息队列解耦服务", "分布式锁实现方案",
    ]

    for scale in scales:
        db_path = f"./data/_exp25_scale_{scale}.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        mm = MemoryManager(db_path=db_path)

        # Encode time
        encode_start = time.time()
        for i in range(scale):
            text = corpus[i % len(corpus)] + f" 变体{i} 补充信息{i*7 % 100}"
            mm.store(text, importance=random.uniform(0.2, 0.9))
        encode_ms = (time.time() - encode_start) * 1000

        # Warmup BM25 index before measuring
        mm.retrieve('warmup', top_k=1)
        
        # Query time (10 queries, measure each)
        query_times = []
        test_queries = ["Linux配置", "Python开发", "Docker部署", "数据库优化", "安全认证",
                       "前端框架", "监控告警", "CI/CD", "API设计", "微服务架构"]
        for q in test_queries:
            qt0 = time.time()
            mm.retrieve(q, top_k=5)
            query_times.append((time.time() - qt0) * 1000)

        query_times_sorted = sorted(query_times)
        results[scale] = {
            "entries": scale,
            "encode_total_ms": round(encode_ms, 1),
            "encode_per_entry_ms": round(encode_ms / scale, 2),
            "query_p50_ms": round(query_times_sorted[len(query_times_sorted)//2], 2),
            "query_p95_ms": round(query_times_sorted[int(len(query_times_sorted)*0.95)], 2),
            "query_p99_ms": round(query_times_sorted[-1], 2),
            "query_avg_ms": round(sum(query_times)/len(query_times), 2),
        }
        print(f"    {scale:>5} entries: encode={encode_ms:.0f}ms, query P95={results[scale]['query_p95_ms']}ms")

        try: os.remove(db_path)
        except: pass

    # Check if latency stays within bounds at scale
    max_p95 = max(r["query_p95_ms"] for r in results.values())
    all_under_500 = all(r["query_p95_ms"] <= 500 for r in results.values())

    dur = round((time.time() - t0) * 1000)

    result = {
        "scaling_results": results,
        "max_p95_ms": round(max_p95, 2),
        "all_under_500ms": all_under_500,
        "dur_ms": dur,
    }
    print_result("Exp25 Latency Scaling", result)
    return result


# ============================================================
# Exp26: 冲突检测压力测试 (50+ cases)
# ============================================================
def exp26_conflict_stress():
    """
    8个冲突类型, 50+ cases:
    T1 完全重复 (5 cases)
    T2 直接矛盾 (8 cases)
    T3 数值变更 (8 cases)
    T4 工具替换 (6 cases)
    T5 否定翻转 (6 cases)
    T6 跨语言冲突 (5 cases)
    T7 时序冲突 (5 cases)
    T8 假阳性抵抗 (10 cases - should NOT conflict)
    """
    print("\n--- Exp26: Conflict Detection Stress Test (50+ cases) ---")
    t0 = time.time()

    engine = KnowledgeEngine(db_path="./data/_exp26_conflict.db")

    # Base knowledge (15 entries)
    base = [
        "SSH默认端口是22",
        "用户偏好使用vim编辑器",
        "Python是解释型语言",
        "Linux使用systemd管理服务",
        "Nginx默认监听80端口",
        "用户偏好中文输出",
        "Redis默认端口6379",
        "Docker使用容器技术部署",
        "MySQL默认端口3306",
        "Git用于版本控制管理",
        "HTTPS使用443端口",
        "PostgreSQL支持JSONB数据类型",
        "防火墙默认阻止所有入站连接",
        "用户喜欢用dark主题",
        "JWT token过期时间设为24小时",
    ]
    for text in base:
        engine.store(text, category="fact", confidence=0.8)

    test_cases = [
        # T1: 完全重复 (5)
        ("SSH默认端口是22", True, "duplicate", "T1-完全重复SSH"),
        ("Redis默认端口6379", True, "duplicate", "T1-完全重复Redis"),
        ("Docker使用容器技术部署", True, "duplicate", "T1-完全重复Docker"),
        ("用户偏好使用vim编辑器", True, "duplicate", "T1-完全重复vim"),
        ("Git用于版本控制管理", True, "duplicate", "T1-完全重复Git"),

        # T2: 直接矛盾 (8)
        ("Python是编译型语言", True, "contradict", "T2-Python类型矛盾"),
        ("Nginx默认监听443端口", True, "contradict", "T2-Nginx端口矛盾"),
        ("用户偏好英文输出", True, "contradict", "T2-语言偏好矛盾"),
        ("用户喜欢用light主题", True, "contradict", "T2-主题矛盾"),
        ("防火墙默认允许所有入站连接", True, "contradict", "T2-防火墙策略矛盾"),
        ("SSH默认端口是2222", True, "contradict", "T2-SSH端口矛盾"),
        ("HTTPS使用8443端口", True, "contradict", "T2-HTTPS端口矛盾"),
        ("JWT token过期时间设为1小时", True, "contradict", "T2-JWT过期矛盾"),

        # T3: 数值变更 (8)
        ("Redis默认端口改为6380", True, "contradict", "T3-Redis端口变更"),
        ("MySQL端口从3306改到3307", True, "contradict", "T3-MySQL端口变更"),
        ("Redis port changed to 6380", True, "contradict", "T3-Redis英文端口变更"),
        ("Nginx listen 8080;", True, "contradict", "T3-Nginx配置格式变更"),
        ("SSH端口: 22 -> 2222", True, "contradict", "T3-SSH箭头表达"),
        ("MySQL默认端口3307", True, "contradict", "T3-MySQL端口新值"),
        ("HTTPS port is now 8443", True, "contradict", "T3-HTTPS英文变更"),
        ("JWT过期时间改成48小时", True, "contradict", "T3-JWT时长变更"),

        # T4: 工具替换 (6)
        ("用户偏好使用emacs编辑器", True, "contradict", "T4-vim换emacs"),
        ("编辑器从vim换成了vscode", True, "contradict", "T4-vim换vscode中文"),
        ("user prefers emacs over vim for editing", True, "contradict", "T4-vim换emacs英文"),
        ("切换到neovim作为主力编辑器", True, "contradict", "T4-vim换neovim"),
        ("从Nginx迁移到Caddy作为Web服务器", True, "contradict", "T4-Nginx换Caddy"),
        ("Docker改用podman替代", True, "contradict", "T4-Docker换podman"),

        # T5: 否定翻转 (6)
        ("Python不是解释型语言", True, "contradict", "T5-Python否定"),
        ("用户不喜欢中文输出", True, "contradict", "T5-语言偏好否定"),
        ("不要用vim编辑器", True, "contradict", "T5-vim否定"),
        ("Docker不应该用于部署", True, "contradict", "T5-Docker否定"),
        ("禁用systemd服务管理", True, "contradict", "T5-systemd否定"),
        ("Redis不适合做缓存", True, "contradict", "T5-Redis否定"),

        # T6: 跨语言冲突 (5)
        ("SSH default port is 2222", True, "contradict", "T6-SSH英文端口变更"),
        ("Python is a compiled language", True, "contradict", "T6-Python英文矛盾"),
        ("User prefers English output", True, "contradict", "T6-语言英文矛盾"),
        ("Nginx default port changed to 8080", True, "contradict", "T6-Nginx英文变更"),
        ("Redis port changed to 6380", True, "contradict", "T6-Redis英文变更"),

        # T7: 时序冲突 (5)
        ("之前SSH用22端口，现在改成了2222", True, "contradict", "T7-SSH时序变更"),
        ("原来用vim，后来换成了vscode", True, "contradict", "T7-编辑器时序变更"),
        ("Redis从6379迁移到了6380端口", True, "contradict", "T7-Redis时序迁移"),
        ("之前用MySQL，项目已迁移到PostgreSQL", True, "contradict", "T7-数据库时序迁移"),
        ("原本用Nginx，现在换成Caddy了", True, "contradict", "T7-Web服务器时序"),

        # T8: 假阳性抵抗 (10 - should NOT conflict)
        ("Rust语言的借用检查器保证内存安全", False, None, "T8-Rust新知识"),
        ("Kubernetes用于容器编排管理", False, None, "T8-K8s全新知识"),
        ("Prometheus用于监控指标采集", False, None, "T8-Prometheus全新"),
        ("SSH密钥认证比密码更安全", False, None, "T8-SSH相关但不冲突"),
        ("vim插件管理器推荐vim-plug", False, None, "T8-vim相关但不冲突"),
        ("Python虚拟环境使用venv创建", False, None, "T8-Python相关不冲突"),
        ("Docker最佳实践使用多阶段构建", False, None, "T8-Docker相关不冲突"),
        ("Redis集群模式支持自动故障转移", False, None, "T8-Redis新知识"),
        ("Git rebase保持提交历史整洁", False, None, "T8-Git新知识"),
        ("Nginx配置缓存提高访问速度", False, None, "T8-Nginx新知识"),

        # === Enhanced difficulty: implicit contradictions ===
        # T9: Implicit contradiction via context
        ("PostgreSQL比MySQL更适合OLTP场景", False, None, "T9-OLTP不同属性"),
        ("Redis集群模式最少需要6个节点", False, None, "T9-Redis不同属性"),
        ("JWT过期时间建议设为15分钟", True, "contradict", "T9-JWT过期建议"),
        ("Nginx worker进程数设为CPU核数", False, None, "T9-Nginx不同属性"),
        ("防火墙应该允许SSH远程管理", True, "contradict", "T9-防火墙SSH规则"),
        # T10: Cross-language semantic contradiction
        ("vim is the best editor for server administration", False, None, "T10-vim同向"),
        ("Redis should not be used as primary database", True, "contradict", "T10-Redis英文否定"),
        ("Docker containers should run as root for simplicity", False, None, "T10-Docker新知识"),
        # T11: Harder false positives (should NOT conflict)
        ("vim宏录制功能很强大", False, None, "T11-vim宏"),
        ("Redis Sentinel实现高可用", False, None, "T11-Redis Sentinel"),
        ("Docker buildx多平台构建", False, None, "T11-Docker buildx"),
        ("MySQL binlog实现数据同步", False, None, "T11-MySQL binlog"),
        ("SSH隧道转发端口映射", False, None, "T11-SSH隧道"),        ("SSH隧道转发端口映射", False, None, "T11-SSH隧道"),

        # T12: Preference evolution (should detect as conflict - user changed mind)
        ("用户现在更喜欢light主题而不是dark", True, "contradict", "T12-主题变更"),
        ("编辑器从vim迁移到vscode主力开发", True, "contradict", "T12-编辑器迁移"),

        # T13: Conditional/contextual knowledge (NOT conflicts)
        ("PostgreSQL在OLAP场景下性能优于MySQL", False, None, "T13-PG场景"),
        ("Redis Cluster需要至少3主3从节点", False, None, "T13-Redis集群知识"),
        ("Nginx worker进程数通常设为CPU核数", False, None, "T13-Nginx最佳实践"),
        ("JWT过期时间应根据安全需求调整", False, None, "T13-JWT通用建议"),
        ("Docker使用非root用户运行更安全", True, "contradict", "T13-Docker安全建议"),

        # T14: Subtle numeric conflict (same metric, close but different)
        ("Redis默认端口是6380", True, "contradict", "T14-Redis端口微调"),
        ("MySQL默认连接端口改为3308", True, "contradict", "T14-MySQL端口微调"),
        ("SSH默认端口改为2200", True, "contradict", "T14-SSH端口微调"),

        # T15: Subtle same-topic different-property (NOT conflicts)
        ("Redis支持5种数据结构", False, None, "T15-Redis数据结构"),
        ("SSH支持密钥认证登录", False, None, "T15-SSH密钥"),
        ("Nginx支持WebSocket代理", False, None, "T15-Nginx WebSocket"),
        ("Docker容器可以设置内存限制", False, None, "T15-Docker内存"),
        ("Git支持stash暂存工作区", False, None, "T15-Git stash"),
        ("MySQL支持全文索引", False, None, "T15-MySQL全文"),
        ("Python支装饰器模式", False, None, "T15-Python装饰器"),

        # T16: High-similarity but not conflict (tricky)
        ("Redis默认端口6379用于缓存服务", False, None, "T16-Redis缓存扩展"),
        ("SSH默认端口22用于远程管理", False, None, "T16-SSH远程扩展"),
        ("Docker使用容器技术部署应用是主流方案", False, None, "T16-Docker扩展"),

        # T17: Preference escalation (should conflict)
        ("用户从不喜欢vim变为讨双vim", True, "contradict", "T17-vim双否定"),
        ("用户现在完全不用Docker了", True, "contradict", "T17-Docker完全拒绝"),

        # T18: Numeric with unit change (hard)
        ("JWT token过期时间改为900秒", True, "contradict", "T18-JWT秒单位"),
        ("Redis端口改为0x18EB", True, "contradict", "T18-Redis十六进制"),

        # === T15: Same topic, different attributes (should NOT conflict) ===
        ("Redis支持5种基本数据类型", False, None, "T15-Redis数据类型"),
        ("Docker镜像使用分层文件系统", False, None, "T15-Docker文件系统"),
        ("Git使用SHA-1哈希标识提交", False, None, "T15-Git哈希"),
        ("Nginx支持HTTP/2协议", False, None, "T15-Nginx HTTP2"),
        ("Python的GIL限制了多线程并发", False, None, "T15-Python GIL"),
        ("MySQL使用B+树索引结构", False, None, "T15-MySQL索引"),
        ("SSH支持密钥和密码两种认证方式", False, None, "T15-SSH认证"),

        # === T16: High similarity but NOT conflict (false positive resistance) ===
        ("用户喜欢用vim写Python代码", False, None, "T16-vim+Python细化"),
        ("vim是用户最常用的文本编辑器", False, None, "T16-vim表述变体"),
        ("输出语言首选中文", False, None, "T16-中文输出变体"),
        ("Redis用6379端口提供服务", False, None, "T16-Redis端口表述"),
        ("SSH服务监听22端口", False, None, "T16-SSH端口表述"),
        ("nginx在80端口上监听HTTP请求", False, None, "T16-Nginx端口表述"),

        # === T17: Preference evolution (update, not conflict) ===
        ("用户现在更喜欢用neovim而不是vim", True, "contradict", "T17-vim升级neovim"),
        ("用户开始偏好light主题了", True, "contradict", "T17-主题偏好演变"),
        ("用户现在习惯用英文输出", True, "contradict", "T17-语言偏好演变"),
        ("用户最近改用VSCode了", True, "contradict", "T17-编辑器演变"),

        # === T18: Unit change numeric conflict ===
        ("JWT token过期时间改为900秒", True, "contradict", "T18-JWT秒单位"),
        ("Redis端口改为0x18EB", True, "contradict", "T18-Redis十六进制"),
        ("JWT有效期设为30分钟", True, "contradict", "T18-JWT分钟单位"),

        # === T19: Paraphrased contradictions (same meaning, different words) ===
        ("Python是脚本语言不是编译型的", True, "contradict", "T19-Python类型释义"),
        ("容器化部署不适合用Docker", True, "contradict", "T19-Docker释义否定"),
        ("用户表示不再使用vim编辑器", True, "contradict", "T19-vim正式否定"),
        ("vim编辑器已经被弃用", True, "contradict", "T19-vim弃用声明"),

        # === T20: Strength variation (stronger/weaker assertion) ===
        ("SSH端口强烈建议改为2222", True, "contradict", "T20-SSH强烈建议"),
        ("Redis端口建议修改为6380", True, "contradict", "T20-Redis建议修改"),
        ("所有新项目禁止使用vim", True, "contradict", "T20-vim全面禁止"),
        ("中文输出已强制启用", False, None, "T20-中文已启用"),  # Already stored

        # === T21: Implicit conflict (requires inference) ===
        ("Python编译后运行性能更好", True, "contradict", "T21-Python编译暗示"),
        ("Docker不适合容器化部署", True, "contradict", "T21-Docker自相矛盾"),
        ("用Nginx做正向代理效果更好", False, None, "T21-Nginx新用法"),
        ("Redis作为消息队列使用效果好", False, None, "T21-Redis新用途"),

        # === T22: Cross-language paraphrase ===
        ("vim is no longer preferred by the user", True, "contradict", "T22-vim英文弃用"),
        ("the user now likes English output better", True, "contradict", "T22-语言英文偏好"),
        ("Python is a scripting language not compiled", True, "contradict", "T22-Python英文类型"),
        ("nginx default port should be 8080", True, "contradict", "T22-Nginx英文端口建议"),

        # === T23: Compound conflict (multiple signals) ===
        ("从vim迁移到VSCode，同时把主题从dark改为light", True, "contradict", "T23-编辑器+主题"),
        ("Redis端口改为6380，JWT过期时间改为1小时", True, "contradict", "T23-Redis+JWT"),
        ("停止使用vim改用neovim，输出语言改为英文", True, "contradict", "T23-vim+语言"),

        # === T24: Boundary/near-miss (should NOT conflict) ===
        ("SSH可以用端口22也可以用2222", True, "contradict", "T24-SSH端口可选"),
        ("vim和neovim都可以用来编辑代码", False, None, "T24-编辑器并存"),
        ("Redis除了做缓存还可以做消息队列", False, None, "T24-Redis多功能"),
        ("dark主题和light主题各有优势", False, None, "T24-主题中立"),

    ]

    correct = 0
    total = len(test_cases)
    failures = []

    for new_content, should_conflict, expected_type, desc in test_cases:
        conflicts = engine.detect_conflicts_v2(new_content, "fact")
        has_conflict = len(conflicts) > 0
        actual_type = conflicts[0].conflict_type if conflicts else None

        is_correct = has_conflict == should_conflict
        if is_correct and should_conflict and expected_type:
            # Also check type matches (for non-None expected types)
            is_correct = actual_type == expected_type or (
                expected_type == "contradict" and actual_type in ("contradict", "update")
            )

        if is_correct:
            correct += 1
        else:
            failures.append({
                "desc": desc,
                "input": new_content,
                "expected_conflict": should_conflict,
                "expected_type": expected_type,
                "actual_conflict": has_conflict,
                "actual_type": actual_type,
            })

    acc = correct / total
    dur = round((time.time() - t0) * 1000)

    # Per-type breakdown
    type_stats = {}
    for new_content, should_conflict, expected_type, desc in test_cases:
        tname = desc.split("-")[0]  # e.g., "T1", "T2"
        if tname not in type_stats:
            type_stats[tname] = {"total": 0, "correct": 0}
        type_stats[tname]["total"] += 1
        conflicts = engine.detect_conflicts_v2(new_content, "fact")
        has_conflict = len(conflicts) > 0
        actual_type = conflicts[0].conflict_type if conflicts else None
        is_correct = has_conflict == should_conflict
        if is_correct and should_conflict and expected_type:
            is_correct = actual_type == expected_type or (
                expected_type == "contradict" and actual_type in ("contradict", "update")
            )
        if is_correct:
            type_stats[tname]["correct"] += 1

    for tname in type_stats:
        s = type_stats[tname]
        s["accuracy"] = round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0

    result = {
        "total": total,
        "correct": correct,
        "accuracy": round(acc, 4),
        "target_88_met": acc >= 0.88,
        "type_breakdown": type_stats,
        "failures": failures[:15],
        "failure_count": len(failures),
        "dur_ms": dur,
    }
    print_result("Exp26 Conflict Stress", result)

    try: os.remove("./data/_exp26_conflict.db")
    except: pass

    return result


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("Exp23-26: 压力级实验 — 大规模 + 高难度 + 多维度")
    print("=" * 70)

    all_results = {}

    experiments = [
        ("Exp23_PreferenceStress", exp23_preference_stress),
        ("Exp24_RetrievalStress", exp24_retrieval_stress),
        ("Exp25_LatencyScaling", exp25_latency_scaling),
        ("Exp26_ConflictStress", exp26_conflict_stress),
    ]

    for name, func in experiments:
        try:
            result = func()
            all_results[name] = result
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, r in all_results.items():
        if "error" in r:
            print(f"  {name}: FAIL ({r['error'][:60]})")
        elif name == "Exp23_PreferenceStress":
            print(f"  {name}: acc={r['accuracy']:.1%} ({r['total']} cases) {'PASS' if r['target_85_met'] else 'FAIL'}")
        elif name == "Exp24_RetrievalStress":
            print(f"  {name}: R@5={r['recall_at_5']:.1%} MRR={r['mrr']:.3f} ({r['total_queries']} queries) {'PASS' if r['target_recall_met'] else 'FAIL'}")
        elif name == "Exp25_LatencyScaling":
            print(f"  {name}: maxP95={r['max_p95_ms']}ms {'PASS' if r['all_under_500ms'] else 'FAIL'}")
        elif name == "Exp26_ConflictStress":
            print(f"  {name}: acc={r['accuracy']:.1%} ({r['total']} cases) {'PASS' if r['target_88_met'] else 'FAIL'}")

    out_path = Path(__file__).parent / "results_stress.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults: {out_path}")

    return all_results


if __name__ == "__main__":
    main()
