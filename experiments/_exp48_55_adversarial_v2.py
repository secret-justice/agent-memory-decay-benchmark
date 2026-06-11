# -*- coding: utf-8 -*-
"""Exp48-55: Adversarial Stress Test v2 - Much harder experiments"""
import sys, os, time, json, random, threading, statistics, tempfile, re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
random.seed(42)

RESULTS = {}


def _run(name, tests, threshold):
    passed = sum(1 for _, ok, _ in tests if ok)
    total = len(tests)
    acc = passed / total if total else 0
    result = {
        "total": total,
        "passed": passed,
        "accuracy": round(acc, 4),
        "target_met": acc >= threshold,
        "details": tests,
    }
    RESULTS[name] = result
    print(f"\n=== {name} ===")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result


def _tmp():
    return tempfile.mktemp(suffix=".db")


# ============================================================
# Exp48: Adversarial Preference Classification (150+ cases)
# ============================================================

def exp48_adversarial_preference():
    """7-tier adversarial preference classification test."""
    from src.extraction.zero_llm_pipeline import ZeroLLMPipeline
    print("\n--- Exp48: Adversarial Preference Classification ---")
    pipe = ZeroLLMPipeline()
    T = []

    # L1 (20): Temporal shift - using actual Chinese text
    L1 = [
        ("以前用vim现在改用VSCode了", "preference"),
        ("之前一直用MySQL，最近换成了PostgreSQL", "preference"),
        ("早些年用Python2，现在全部迁移到Python3了", "preference"),
        ("原来用Eclipse，上个月开始改用IntelliJ", "preference"),
        ("之前用SVN管理代码，后来改用Git了", "preference"),
        ("以前部署用Docker，现在改用Podman", "preference"),
        ("之前一直用npm，最近切换到pnpm了", "preference"),
        ("从React转到Vue了，感觉更顺手", "preference"),
        ("去年还在用Webpack，今年全部用Vite了", "preference"),
        ("以前用Postman，现在换成Hoppscotch了", "preference"),
        ("之前偏好Ubuntu，最近开始用Debian了", "preference"),
        ("从Maven迁移到Gradle了", "preference"),
        ("原来用Jenkins，后来换成GitHub Actions", "preference"),
        ("之前用Redis做缓存，现在改用Memcached", "preference"),
        ("以前用Nginx，现在改用Caddy了", "preference"),
        ("之前一直用Bash，最近切换到Zsh", "preference"),
        ("之前用Jupyter，现在改用VSCode的Notebook", "preference"),
        ("原来用SQLite，项目大了换成PostgreSQL", "preference"),
        ("从jQuery时代到现在用原生JS", "preference"),
        ("以前用FTP传文件，现在改用rsync", "preference"),
    ]

    # L2 (20): Conditional preference
    L2 = [
        ("小项目用Flask大项目用Django", "preference"),
        ("如果是微服务就用Go，单体应用用Python", "preference"),
        ("快速原型用SQLite，生产环境用PostgreSQL", "preference"),
        ("写脚本用Python，写高性能服务用Rust", "preference"),
        ("个人项目用Vim，团队协作用VSCode", "preference"),
        ("本地开发用Docker，生产部署用K8s", "preference"),
        ("测试环境用H2，正式环境用MySQL", "preference"),
        ("静态页面用Hugo，动态站点用Next.js", "preference"),
        ("日志收集用ELK，轻量场景用Loki", "preference"),
        ("前端小项目用Vue，大项目用React", "preference"),
        ("数据量小时用SQLite，量大用ClickHouse", "preference"),
        ("API网关小规模用Nginx，大规模用Kong", "preference"),
        ("CI用GitHub Actions，CD用ArgoCD", "preference"),
        ("实时通信用WebSocket，轮询用SSE", "preference"),
        ("文档写作用Markdown，复杂排版用LaTeX", "preference"),
        ("单元测试用pytest，集成测试用TestContainers", "preference"),
        ("容器编排小规模用Docker Compose，大规模用K8s", "preference"),
        ("消息队列简单用Redis，复杂用Kafka", "preference"),
        ("监控用Prometheus，日志用Grafana Loki", "preference"),
        ("样式简单用Tailwind，复杂交互用Styled Components", "preference"),
    ]

    # L3 (20): Double negation
    L3 = [
        ("不是说不能用Docker，但我觉得Podman更好", "preference"),
        ("不是不想用React，只是Vue更适合我们的场景", "preference"),
        ("并不是说MySQL不好，只是PostgreSQL功能更全", "preference"),
        ("不是不能用Vim写代码，但VSCode效率更高", "preference"),
        ("不是说Java不行，但我更倾向Kotlin", "preference"),
        ("并非不认叨GraphQL，REST对我们来说更简单", "preference"),
        ("不是不喜欢Webpack，Vite确实更快", "preference"),
        ("不是不能用jQuery，只是原生JS已经够用了", "preference"),
        ("不是说CentOS不好，我就是更习惯Ubuntu", "preference"),
        ("不是不想学Emacs，Vim已经够我用了", "preference"),
        ("不是不用SVN，是Git分布式更方便", "preference"),
        ("不是说Angular不行，Vue上手更快", "preference"),
        ("不是不承认Selenium好，Playwright更现代", "preference"),
        ("不是觉得Redis不好，只是Memcached更轻量", "preference"),
        ("不是不接受Docker，K8s对我们太重了", "preference"),
        ("不是说Go不好写，Rust安全性更高", "preference"),
        ("不是不用Postman，curl在终端更方便", "preference"),
        ("不是说Maven不好，Gradle构建更快", "preference"),
        ("不是不用Nginx，Traefik配置更简单", "preference"),
        ("不是说Bash不好，Zsh插件生态更丰富", "preference"),
    ]

    # L4 (20): Sarcasm
    L4 = [
        ("我真是太喜欢vim了，每次卡死都开心", "preference"),
        ("Eclipse真是太好用了，光启动就等5分钟", "preference"),
        ("Jenkins真是太稳定了，每次升级都出bug", "preference"),
        ("Gradle的依赖管理真是太清晰了，冲突100个", "preference"),
        ("Magento真是太轻量了，部署要一天", "preference"),
        ("我可太爱Spring了，配置文件比代码还多", "preference"),
        ("WordPress真是太安全了，每月都有漏洞补丁", "preference"),
        ("Angular的学习曲线真是太友好了，新手一天入门", "preference"),
        ("我就是喜欢Perl，代码可读性简直完美", "preference"),
        ("SVN真是太方便了，合并冲突每次都很快解决", "preference"),
        ("GWT真是太现代了，编译速度飞快", "preference"),
        ("我可太喜欢SOAP了，比REST简洁多了", "preference"),
        ("Struts真是太安全了，从来没有过安全漏洞", "preference"),
        ("VB6真是太好维护了，10年老代码一看就懂", "preference"),
        ("XML配置真是太优雅了，比JSON强多了", "preference"),
        ("我真是太享受Maven的下载速度了", "preference"),
        ("C++的内存管理真是太省心了", "preference"),
        ("PHP的类型系统真是太严谨了", "preference"),
        ("我可太喜欢IE6兼容性调试了", "preference"),
        ("Bash脚本的错误处理真是太优雅了", "preference"),
    ]

    # L5 (20): Buried in long text
    L5 = [
        ("今天和团队开会讨论了下个季度的技术选型问题，前端同事建议继续用React，后端同事觉得Go比Python性能好，数据库方面有人说MongoDB适合我们的场景，也有人坚持用PostgreSQL，最后我拍板说我个人偏好用TypeScript写前端因为类型安全能减少很多bug，所以以后前端项目默认用TypeScript加React", "preference"),
        ("上周去参加了一个技术大会，听了好几场演讲，有关于云原生的，有关于AI的，还有关于区块链的，其中有一场讲Serverless架构的让我印象深刻，回来之后我研究了一下AWS Lambda和Cloudflare Workers，我个人更倾向于使用Cloudflare Workers因为冷启动时间更短，以后serverless项目优先考虑CF Workers", "preference"),
        ("帮我看一下这个项目的技术栈，后端用的是Python Flask框架，数据库是MySQL，缓存是Redis，消息队列用的是RabbitMQ，部署在Docker容器里，我发现Flask在高并发下性能不太行，我决定以后新项目都改用FastAPI因为它支持异步而且性能好很多", "preference"),
        ("最近在做性能优化，测试了好几种方案，用JMeter压测了Nginx和Caddy的性能，发现Nginx在静态资源分发上更快，但是Caddy自动HTTPS配置真的很方便，综合考虑我决定以后反向代理统一用Caddy因为它省去了证书管理的麻烦", "preference"),
        ("昨天花了半天时间对比了三个ORM框架，SQLAlchemy功能最全但是学习曲线陡，Peewee最轻量但是高级功能不够，TortoiseORM支持异步但是生态还不够成熟，权衡之后我觉得小项目用Peewee够了大项目还是得SQLAlchemy", "preference"),
        ("团队最近在讨论代码规范，有人说用ESLint有人说用Prettier，还有人建议两个一起用，我个人一直习惯用ESLint加Prettier的组合因为能自动修复大部分格式问题，以后所有项目都配置这两个工具的组合方案", "preference"),
        ("研究了三种消息队列的方案，Kafka吞吐量最高但是运维复杂，RabbitMQ功能全面但是性能一般，Redis Stream最轻量但是可靠性稍差，根据我们业务特点我偏好使用RabbitMQ因为它的消息确认机制更成熟", "preference"),
        ("做了一个内部的前端框架对比评测，React生态最丰富，Vue上手最快，Svelte性能最好，Angular最完整但最重，经过几周的试用我倾向于在新项目中用Svelte因为它编译后体积小运行快", "preference"),
        ("客户要求我们做一个实时通讯功能，技术选型上可以用WebSocket、SSE、Long Polling，我觉得WebSocket最适合全双工通信场景，所以以后实时功能都默认采用WebSocket方案", "preference"),
        ("在选择监控系统的时候对比了Prometheus、Datadog、Grafana Cloud，Prometheus开源免费但是需要自己运维，Datadog功能全但是贵，Grafana Cloud性价比不错，我最终决定用Prometheus加Grafana自建方案因为长期成本最低", "preference"),
        ("日志系统选型考虑了ELK Stack、Loki、ClickHouse三种方案，ELK最成熟但是资源消耗大，Loki和Grafana配合好但是查询能力有限，ClickHouse查询最快但是运维成本高，我倾向于用Loki因为我们的日志量不算特别大", "preference"),
        ("CI/CD平台选型比较了Jenkins、GitLab CI、GitHub Actions、CircleCI，Jenkins最灵活但维护成本高，GitHub Actions和代码托管无缝集成，我偏好使用GitHub Actions因为配置简单和仓库在一起管理方便", "preference"),
        ("容器编排方案选型讨论了很久，Docker Swarm简单但功能有限，K8s强大但学习曲线陡，Nomad介于两者之间，最终我决定用Kubernetes因为社区最活跃遇到问题容易找到解决方案", "preference"),
        ("项目需要选择一个测试框架，pytest功能最丰富，unittest是标准库，nose2比较小众，我个人一直用pytest因为它的fixture机制和参数化功能非常好用", "preference"),
        ("数据库迁移工具选型，Alembic最成熟，Flyway支持多数据库，Django自带migration，我们的Python项目我偏好用Alembic因为它和SQLAlchemy配合最好", "preference"),
        ("API文档工具选型，Swagger UI最老牌，Redoc更现代，Stoplight最商业，我倾向用Redoc因为它生成的文档页面更美观而且支持OpenAPI 3.0", "preference"),
        ("构建工具选型讨论，Webpack最成熟但配置复杂，Vite最快但生态稍弱，Parcel零配置但灵活性不够，我决定以后前端项目统一用Vite因为开发体验最好", "preference"),
        ("在选型缓存方案，Redis功能最全面，Memcached最简单，Hazelcast支持分布式，我偏好用Redis因为它除了缓存还能做消息队列和排行榜等多种用途", "preference"),
        ("项目需要选一个CSS方案，Tailwind最灵活，Bootstrap最快速，Material UI组件最全，我个人更喜欢用Tailwind CSS因为自定义能力强而且打包体积小", "preference"),
        ("搜索引擎选型，Elasticsearch功能最强，Meilisearch最轻量，Algolia最快但贵，我倾向用Meilisearch因为对于我们的场景它的搜索质量已经够用而且部署简单", "preference"),
    ]

    # L6 (25): Mixed language code-switching
    L6 = [
        ("I've been using vim for years but 最近改用VSCode了", "preference"),
        ("Our team prefers Python over Java, 但是性能敏感的service用Go", "preference"),
        ("After evaluating several options, 我觉得PostgreSQL比MySQL好用", "preference"),
        ("I always use Docker for local dev, 但生产环境用bare metal", "preference"),
        ("My default setup is Neovim + tmux, 编辑器不太重要的时候直接用VSCode", "preference"),
        ("We decided to switch from Jenkins to GitHub Actions, CI/CD流程简化了很多", "preference"),
        ("I prefer functional programming style, 所以写Python也尽量用map/filter/reduce", "preference"),
        ("After trying both React and Vue, 我觉得Vue的template语法更直观", "preference"),
        ("For database, I usually go with PostgreSQL, 除非项目很小就用SQLite", "preference"),
        ("I'm a big fan of Rust for system programming, 但是web后端还是用Python更高效", "preference"),
        ("We migrated from monolith to microservices, 选了Go做后端因为concurrency model好", "preference"),
        ("I always configure ESLint + Prettier, 代码风格一致性比什么都重要", "preference"),
        ("Testing framework-wise, I prefer pytest over unittest, fixture机制太好用了", "preference"),
        ("For deployment, I lean towards Kubernetes, 小项目就Docker Compose搞定", "preference"),
        ("Log aggregation的话我用ELK stack, 但最近在考虑切换到Loki", "preference"),
        ("API design方面我偏好RESTful, 只有复杂查询场景才用GraphQL", "preference"),
        ("I strongly prefer TypeScript over JavaScript, 类型安全能避免很多runtime error", "preference"),
        ("Message queue选择上, 简单场景用Redis Stream, 复杂场景用Kafka", "preference"),
        ("For monitoring, I default to Prometheus + Grafana, 商业方案太贵了", "preference"),
        ("Editor choice: Neovim for quick edits, IntelliJ for Java projects", "preference"),
        ("ORM方面, 我习惯用SQLAlchemy, Django项目就用自带的ORM", "preference"),
        ("Package manager: npm以前用, 现在都换pnpm了因为省空间", "preference"),
        ("I prefer serverless architecture, 但有些场景还是得用traditional server", "preference"),
        ("Container registry的话我用Docker Hub, 企业项目用Harbor自建", "preference"),
        ("For static sites, I always reach for Hugo, 动态内容多的话用Next.js", "preference"),
    ]

    # L7 (25): Implicit from behavior patterns
    L7 = [
        ("每次新项目我都先装好Zsh和oh-my-zsh", "preference"),
        ("我的开发环境标配是VSCode加Vim插件", "preference"),
        ("团队代码审查我要求必须跑pytest", "preference"),
        ("部署流程我已经标准化为Docker加K8s", "preference"),
        ("数据库我一般从PostgreSQL开始评估", "preference"),
        ("写Python的时候我都用type hints加mypy", "preference"),
        ("前端项目我的起步模板是Vite加React加TypeScript", "preference"),
        ("API开发我习惯先写OpenAPI spec再写代码", "preference"),
        ("我的CI模板一直是GitHub Actions", "preference"),
        ("日志方案我推荐客户用ELK", "preference"),
        ("做数据分析我一般用Pandas加Matplotlib", "preference"),
        ("爬虫项目我用Scrapy框架", "preference"),
        ("机器学习项目我的默认框架是PyTorch", "preference"),
        ("微服务通信我推荐gRPC", "preference"),
        ("配置管理我用Terraform", "preference"),
        ("密钥管理我用HashiCorp Vault", "preference"),
        ("文档站点我用Docusaurus", "preference"),
        ("项目管理我用Linear", "preference"),
        ("设计协作我推荐Figma", "preference"),
        ("代码托管我首选GitHub", "preference"),
        ("邮件发送我用SendGrid", "preference"),
        ("文件存储我用MinIO", "preference"),
        ("搜索功能我接入Meilisearch", "preference"),
        ("任务队列我用Celery加Redis", "preference"),
        ("实时推送我用Socket.IO", "preference"),
    ]

    all_cases = L1 + L2 + L3 + L4 + L5 + L6 + L7
    tier_offsets = [0, 20, 40, 60, 80, 100, 125]
    tier_names = ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]

    for i, (text, expected) in enumerate(all_cases):
        result = pipe.classify(text)
        ok = result == expected
        tier = "L7"
        for ti, offset in enumerate(tier_offsets):
            if i < offset:
                break
            tier = tier_names[ti]
        T.append((f"{tier}_case{i:03d}", ok, f"exp={expected} got={result}"))

    return _run("exp48_adversarial_preference", T, 0.75)



# ============================================================
# Exp49: Adversarial Conflict Detection (200+ cases)
# ============================================================

def exp49_adversarial_conflict():
    """10-type adversarial conflict detection test."""
    from src.knowledge.knowledge_engine import KnowledgeEngine
    print("\n--- Exp49: Adversarial Conflict Detection ---")
    db = _tmp()
    engine = KnowledgeEngine(db_path=db)
    T = []

    # T25 (20): Port/service implicit conflict
    T25 = [
        ("SSH runs on port 22", "改为2222端口", True),
        ("Nginx listens on port 80", "改用8080端口", True),
        ("MySQL default port 3306", "迁移到3307", True),
        ("Redis runs on 6379", "改为16379", True),
        ("PostgreSQL on 5432", "改为5433", True),
        ("MongoDB on 27017", "迁移到27018", True),
        ("Elasticsearch on 9200", "改为9201", True),
        ("RabbitMQ on 5672", "调整到5673", True),
        ("Kafka broker on 9092", "改到9093端口", True),
        ("MinIO on 9000", "改为9001", True),
        ("Grafana on 3000", "改用3001", True),
        ("Prometheus on 9090", "迁移到9091", True),
        ("Jupyter on 8888", "改为8889", True),
        ("Tomcat on 8080", "改为8081", True),
        ("FTP on 21", "改用2121端口", True),
        ("SMTP on 25", "改为587端口", True),
        ("DNS on 53", "改到5353端口", True),
        ("Memcached on 11211", "改为11212", True),
        ("Consul on 8500", "改为8501", True),
        ("Vault on 8200", "改为8201", True),
    ]

    # T26 (20): Temporal shift conflict
    T26 = [
        ("之前一直用vim开发", "现在改用VSCode了", True),
        ("去年用React做前端", "今年切换到Vue了", True),
        ("原来部署在AWS上", "最近迁移到阿里云了", True),
        ("之前用MySQL存数据", "上个月切换到PostgreSQL", True),
        ("以前用Jenkins做CI", "现在换成GitHub Actions", True),
        ("之前用SVN管理代码", "目前统一用Git", True),
        ("原来的监控方案是Zabbix", "新方案改用Prometheus", True),
        ("之前用Nginx做反向代理", "现在换成Traefik", True),
        ("以前日志存ES里", "现在改用ClickHouse", True),
        ("之前用Puppet做配置管理", "现在用Ansible", True),
        ("之前用Heroku部署", "现在迁移到Railway", True),
        ("以前用MongoDB", "后来发现PostgreSQL更合适就换了", True),
        ("之前的测试框架是unittest", "后来改用pytest", True),
        ("原来用Cordova做移动端", "现在换Flutter", True),
        ("之前用Bower管理前端依赖", "现在用npm", True),
        ("以前用Gulp做构建", "现在用Vite", True),
        ("之前用PhantomJS做测试", "现在用Playwright", True),
        ("之前的缓存用Memcached", "后来换成Redis", True),
        ("以前用FTP部署代码", "现在用CI/CD自动部署", True),
        ("之前用XML做配置", "现在改用YAML", True),
    ]

    # T27 (20): Scope limitation conflict
    T27 = [
        ("所有项目都用Python开发", "前端项目用JavaScript不用Python", True),
        ("整个公司统一用MySQL", "数据仓库项目用ClickHouse", True),
        ("全栈都用TypeScript", "机器学习模块用Python", True),
        ("所有服务都部署在K8s上", "数据库服务不在K8s里运行", True),
        ("所有API都用REST", "实时通知接口用WebSocket", True),
        ("统一用GitLab管理代码", "开源项目放在GitHub上", True),
        ("全部日志发到ELK", "安全审计日志单独存Splunk", True),
        ("所有测试用pytest", "E2E测试用Playwright", True),
        ("统一用Docker打包", "GPU服务直接部署不用容器", True),
        ("所有配置用YAML", "敏感配置用Vault管理", True),
        ("全部用Nginx做代理", "gRPC服务用Envoy", True),
        ("统一用Prometheus监控", "业务指标用DataDog", True),
        ("所有静态资源上CDN", "内网资源不上CDN", True),
        ("全部用Redis做缓存", "大文件缓存用本地磁盘", True),
        ("统一用HTTPS", "内部服务间用mTLS", True),
        ("所有邮件走SendGrid", "营销邮件走Mailchimp", True),
        ("全部用Jira做项目管理", "开源社区用GitHub Issues", True),
        ("统一用Figma做设计", "流程图用draw.io", True),
        ("所有文档用Markdown", "合同文档用Word", True),
        ("统一用Slack沟通", "客户沟通用邮件", True),
    ]

    # T28 (20): Numeric drift conflict
    T28 = [
        ("Redis最大内存4GB", "改为8GB", True),
        ("MySQL连接池最大100", "改为200", True),
        ("Nginx worker进程数4", "改为8", True),
        ("JVM堆内存2G", "调到4G", True),
        ("TCP超时时间30秒", "改为60秒", True),
        ("日志保留7天", "改为30天", True),
        ("JWT过期时间2小时", "改为4小时", True),
        ("缓存TTL为300秒", "改为600秒", True),
        ("批量插入大小1000", "改为5000", True),
        ("线程池核心大小10", "改为20", True),
        ("请求限流100次/分钟", "改为200次/分钟", True),
        ("数据库连接超时5秒", "改为10秒", True),
        ("消息队列prefetch 10", "改为50", True),
        ("备份保留5份", "改为10份", True),
        ("索引分片数3", "改为5", True),
        ("副本数2", "改为3", True),
        ("最大上传文件50MB", "改为100MB", True),
        ("WebSocket心跳30秒", "改为15秒", True),
        ("重试次数3次", "改为5次", True),
        ("并发请求数50", "改为100", True),
    ]


    # T29 (20): Partial update conflict
    T29 = [
        ("服务端口8080，使用HTTP", "端口改为8443，使用HTTPS", True),
        ("数据库读写分离，主库MySQL", "主库改为PostgreSQL", True),
        ("前端用React加Redux", "状态管理改为Zustand", True),
        ("后端用Flask加Gunicorn", "WSGI改为Uvicorn", True),
        ("用Docker加docker-compose部署", "改为Docker加K8s", True),
        ("监控用Prometheus加Grafana", "告警改为Alertmanager", True),
        ("日志用Filebeat加ES", "传输改为Fluentd", True),
        ("用Celery加RabbitMQ做任务队列", "Broker改为Redis", True),
        ("认证用JWT加Redis", "Token存储改为数据库", True),
        ("搜索用ES加Kibana", "可视化改为Grafana", True),
        ("用Nginx加Gunicorn", "应用服务器改为uWSGI", True),
        ("用Terraform加AWS", "云平台改为Azure", True),
        ("用Ansible加CentOS", "系统改为Ubuntu", True),
        ("用GitLab CI加Docker", "镜像仓库改为Harbor", True),
        ("用PyTorch加CUDA11", "CUDA改为12", True),
        ("用Pandas加NumPy处理数据", "NumPy改为Polars", True),
        ("用FastAPI加Pydantic v1", "Pydantic改为v2", True),
        ("用Selenium加Chrome", "浏览器改为Firefox", True),
        ("用Webpack加Babel", "编译器改为SWC", True),
        ("用GraphQL加Apollo", "客户端改为Relay", True),
    ]

    # T30 (20): Near-duplicate NOT conflict (should NOT detect)
    T30 = [
        ("Python是项目的主要开发语言", "Python是团队的主力编程语言", False),
        ("用Docker容器化部署服务", "使用Docker来容器化应用", False),
        ("Nginx作为反向代理使用", "Nginx用于反向代理", False),
        ("PostgreSQL存储业务数据", "PostgreSQL用来保存业务数据", False),
        ("Redis提供缓存加速", "Redis用于缓存提升性能", False),
        ("GitHub Actions做CI/CD", "GitHub Actions用于持续集成", False),
        ("Prometheus采集监控指标", "Prometheus负责监控数据采集", False),
        ("Elasticsearch做全文搜索", "Elasticsearch用于全文检索", False),
        ("Kafka处理消息队列", "Kafka作为消息中间件使用", False),
        ("Grafana展示监控面板", "Grafana用来做监控可视化", False),
        ("Terraform管理云资源", "Terraform用于基础设施管理", False),
        ("Jest测试前端组件", "Jest用于前端单元测试", False),
        ("Flask提供REST API", "Flask用来构建RESTful接口", False),
        ("Vue开发管理后台", "Vue用于构建管理后台界面", False),
        ("MinIO提供对象存储", "MinIO用于文件对象存储", False),
        ("Consul做服务发现", "Consul负责微服务发现", False),
        ("Loki收集日志", "Loki用于日志聚合", False),
        ("ArgoCD做GitOps部署", "ArgoCD用于GitOps流水线", False),
        ("Envoy做API网关", "Envoy作为API网关使用", False),
        ("Vault管理密钥", "Vault用于密钥和证书管理", False),
    ]


    # T31 (20): Cross-domain conflict
    T31 = [
        ("前端用React框架", "前端改为Vue框架", True),
        ("后端用Python语言", "后端改为Go语言", True),
        ("数据库用MySQL", "数据库改为MongoDB", True),
        ("缓存用Redis", "缓存改为Memcached", True),
        ("搜索用ES", "搜索改为Solr", True),
        ("消息队列用Kafka", "队列改为RabbitMQ", True),
        ("容器用Docker", "容器改为Podman", True),
        ("编排用K8s", "编排改为Docker Swarm", True),
        ("CI用Jenkins", "CI改为GitLab CI", True),
        ("监控用Prometheus", "监控改为Datadog", True),
        ("日志用ELK", "日志改为Loki", True),
        ("配置用Ansible", "配置改为Puppet", True),
        ("网关用Nginx", "网关改为Kong", True),
        ("存储用S3", "存储改为MinIO", True),
        ("DNS用Cloudflare", "DNS改为AWS Route53", True),
        ("CDN用CloudFront", "CDN改为Akamai", True),
        ("邮件用SendGrid", "邮件改为SES", True),
        ("支付用Stripe", "支付改为PayPal", True),
        ("认证用Auth0", "认证改为Keycloak", True),
        ("文档用Swagger", "文档改为Redoc", True),
    ]

    # T32 (20): Encoded negation conflict
    T32 = [
        ("启用SSL证书验证", "禁用SSL证书验证", True),
        ("开启Gzip压缩", "关闭Gzip压缩", True),
        ("允许跨域请求", "禁止跨域请求", True),
        ("启用数据库慢查询日志", "禁用慢查询日志", True),
        ("开启自动备份", "关闭自动备份", True),
        ("允许远程连接", "禁止远程连接", True),
        ("启用写入确认", "关闭写入确认", True),
        ("开启健康检查", "关闭健康检查", True),
        ("启用请求缓存", "禁用请求缓存", True),
        ("允许匿名访问", "禁止匿名访问", True),
        ("启用日志轮转", "禁用日志轮转", True),
        ("开启连接池", "关闭连接池", True),
        ("启用TLS加密", "禁用TLS加密", True),
        ("开启索引优化", "关闭索引优化", True),
        ("允许批量操作", "禁止批量操作", True),
        ("启用读写分离", "禁用读写分离", True),
        ("开启数据压缩", "关闭数据压缩", True),
        ("启用自动重试", "禁用自动重试", True),
        ("开启内存交换", "关闭内存交换", True),
        ("启用访问日志", "禁用访问日志", True),
    ]


    # T33 (20): Fuzzy semantic conflict
    T33 = [
        ("服务器响应时间目标100ms以内", "响应时间调整到200ms以内", True),
        ("可用性目标99.9%", "可用性提升到99.99%", True),
        ("每日凌晨3点备份", "备份时间改为凌晨4点", True),
        ("使用master-slave架构", "改为primary-standby架构", True),
        ("代码review需要2人批准", "review改为1人批准即可", True),
        ("每周五发版", "发版改为每周三", True),
        ("测试覆盖率要求80%", "覆盖率提高到90%", True),
        ("API版本用v1前缀", "版本改为v2前缀", True),
        ("使用递增ID作为主键", "主键改为UUID", True),
        ("密码最少8位", "密码最少改为12位", True),
        ("Session有效期30分钟", "Session改为60分钟", True),
        ("日志级别设为INFO", "日志级别改为WARN", True),
        ("请求body最大1MB", "body改为10MB", True),
        ("数据库连接超时5秒", "超时改为3秒", True),
        ("CORS允许所有域名", "CORS改为只允许指定域名", True),
        ("缓存策略LRU", "缓存改为LFU", True),
        ("序列化用JSON", "序列化改为Protocol Buffers", True),
        ("负载均衡策略轮询", "改为加权轮询", True),
        ("部署模式蓝绿", "改为金丝雀发布", True),
        ("数据格式UTF-8", "改为GBK", True),
    ]

    # T34 (20): Version constraint conflict
    T34 = [
        ("使用Python 3.8", "升级到Python 3.12", True),
        ("Node.js版本16", "Node.js升级到20", True),
        ("React 17", "升级到React 18", True),
        ("Vue 2", "迁移到Vue 3", True),
        ("Django 3.2", "升级到Django 5.0", True),
        ("Spring Boot 2.x", "升级到Spring Boot 3", True),
        ("Kubernetes 1.24", "升级到1.29", True),
        ("PostgreSQL 13", "升级到PostgreSQL 16", True),
        ("Redis 6", "升级到Redis 7", True),
        ("Elasticsearch 7.x", "升级到ES 8.x", True),
        ("Terraform 1.0", "升级到Terraform 1.7", True),
        ("Ansible 2.9", "升级到Ansible 8", True),
        ("Go 1.18", "升级到Go 1.22", True),
        ("Rust 1.60", "升级到Rust 1.77", True),
        ("TypeScript 4.x", "升级到TypeScript 5.4", True),
        ("Next.js 12", "升级到Next.js 14", True),
        ("Tailwind 2.x", "升级到Tailwind 3", True),
        ("Webpack 4", "升级到Webpack 5", True),
        ("Pydantic v1", "升级到Pydantic v2", True),
        ("CUDA 11", "升级到CUDA 12", True),
    ]

    all_groups = [
        ("T25", T25), ("T26", T26), ("T27", T27), ("T28", T28),
        ("T29", T29), ("T30", T30), ("T31", T31), ("T32", T32),
        ("T33", T33), ("T34", T34),
    ]

    for group_name, cases in all_groups:
        for idx, (existing_text, new_text, expect_conflict) in enumerate(cases):
            engine.store(content=existing_text, category="fact",
                         source=f"test_{group_name}", confidence=0.8)
            conflicts = engine.detect_conflicts_v2(new_text)
            has_conflict = len(conflicts) > 0
            ok = has_conflict == expect_conflict
            T.append((
                f"{group_name}_{idx:02d}",
                ok,
                f"expect={expect_conflict} got={has_conflict} n={len(conflicts)}"
            ))

    return _run("exp49_adversarial_conflict", T, 0.80)



# ============================================================
# Exp50: Extreme Scale Latency (P95 <= 500ms at 20K)
# ============================================================

def exp50_extreme_scale():
    """Latency benchmark at 1K/5K/10K/20K scale with raw SQL batch insert."""
    from src.memory.memory_manager import MemoryManager
    import sqlite3, uuid as _uuid
    print("\n--- Exp50: Extreme Scale Latency ---")
    T = []
    scales = [1000, 5000, 10000, 20000]
    queries = [
        "Python Docker部署", "Redis缓存配置", "Nginx反向代理设置",
        "PostgreSQL数据库优化", "Kubernetes集群管理", "React前端框架",
        "JWT认证方案", "日志收集方案", "CI/CD流水线配置", "微服务架构设计",
        "Elasticsearch搜索优化", "RabbitMQ消息队列", "MySQL主从复制",
        "Ansible自动化运维", "Terraform基础设施", "Prometheus监控告警",
        "GraphQL API设计", "gRPC服务通信", "Docker Compose编排",
        "TypeScript类型系统", "Vue组件开发", "Flask REST API",
        "Celery异步任务", "Redis集群部署", "MongoDB索引优化",
        "Nginx负载均衡", "Gunicorn进程管理",
    ]
    templates = [
        "在{action}{env}环境中使用{tool}进行{aspect}优化，参数设置为{val}，版本{ver}",
        "{action}时{tool}的{aspect}配置：env={env}, val={val}, ver={ver}",
        "{env}环境{aspect}相关：{tool} {action} val={val} ver={ver}",
    ]
    tools = ['Docker', 'Redis', 'Nginx', 'PostgreSQL', 'Kubernetes', 'React',
             'JWT', 'ELK', 'Jenkins', 'Kong', 'ES', 'RabbitMQ', 'MySQL',
             'Ansible', 'Terraform', 'Prometheus', 'GraphQL', 'gRPC',
             'Compose', 'TypeScript', 'Vue', 'Flask', 'Celery', 'MongoDB']
    actions = ['部署', '配置', '监控', '备份', '迁移', '优化', '调试', '升级']
    envs = ['生产', '测试', '开发', '预发布', '灰度']
    aspects = ['性能', '安全', '可用性', '扩展性', '稳定性']
    vals = ['100', '200', '500', '1000', '默认', '自定义', '最大', '最小']
    vers = ['1.0', '2.0', '3.0', '4.0', '5.0', 'latest', 'stable', 'beta']

    for scale in scales:
        dbp = _tmp()
        mm = MemoryManager(db_path=dbp)
        conn = sqlite3.connect(dbp)
        now = time.time()
        batch = []
        for i in range(scale):
            txt = random.choice(templates).format(
                tool=random.choice(tools), action=random.choice(actions),
                env=random.choice(envs), aspect=random.choice(aspects),
                val=random.choice(vals), ver=random.choice(vers))
            mid = str(_uuid.uuid4())
            batch.append((mid, txt, 'knowledge', 0.5,
                          now - random.uniform(0, 86400*30),
                          now - random.uniform(0, 86400)))
        conn.executemany(
            "INSERT INTO memories (id,content,category,importance,level,"
            "access_count,created_at,last_accessed,ttl) "
            "VALUES (?,?,?,?,'stm',0,?,?,-1)",
            [(b[0], b[1], b[2], b[3], b[4], b[5]) for b in batch])
        conn.commit()
        conn.close()
        mm2 = MemoryManager(db_path=dbp)
        _ = mm2.retrieve('warmup', top_k=1)
        latencies = []
        for q in queries:
            t0 = time.perf_counter()
            _ = mm2.retrieve(q, top_k=5)
            latencies.append((time.perf_counter() - t0) * 1000)
        latencies.sort()
        p50 = latencies[len(latencies)//2]
        p95 = latencies[int(len(latencies)*0.95)]
        p99 = latencies[int(len(latencies)*0.99)]
        mx = latencies[-1]
        T.append((f"scale_{scale}", p95 <= 500,
                  f"P50={p50:.1f}ms P95={p95:.1f}ms P99={p99:.1f}ms max={mx:.1f}ms"))
        try:
            os.unlink(dbp)
        except OSError:
            pass

    return _run("exp50_extreme_scale", T, 0.75)

# ============================================================
# Exp51: Adversarial Retrieval (100+ queries)
# ============================================================

def exp51_adversarial_retrieval():
    """Store known facts, then query with adversarial variants."""
    from src.memory.memory_manager import MemoryManager
    print("\n--- Exp51: Adversarial Retrieval ---")
    db = _tmp()
    mm = MemoryManager(db_path=db)
    T = []

    facts = [
        ("Python项目使用Flask框架开发REST API", "preference"),
        ("Docker容器的默认网络模式是bridge", "knowledge"),
        ("PostgreSQL支持JSON数据类型存储", "knowledge"),
        ("Redis缓存过期策略使用LRU淘汰", "preference"),
        ("Nginx配置worker进程数为CPU核心数", "knowledge"),
        ("Git使用rebase代替merge保持线性历史", "preference"),
        ("Kubernetes Pod是最小部署单元", "knowledge"),
        ("JWT Token有效期设置为2小时", "knowledge"),
        ("MySQL默认隔离级别是REPEATABLE READ", "knowledge"),
        ("Prometheus使用Pull模式采集指标", "knowledge"),
        ("Elasticsearch使用倒排索引实现搜索", "knowledge"),
        ("RabbitMQ支持消息确认机制保证可靠性", "knowledge"),
        ("TypeScript编译为JavaScript运行", "knowledge"),
        ("Vue3的Composition API比Options API灵活", "preference"),
        ("GraphQL按需查询减少数据传输量", "knowledge"),
        ("gRPC使用Protocol Buffers序列化", "knowledge"),
        ("Ansible使用YAML编写Playbook", "knowledge"),
        ("Terraform使用HCL描述基础设施", "knowledge"),
        ("Consul提供服务发现和健康检查", "knowledge"),
        ("Vault支持动态密钥生成", "knowledge"),
        ("Loki只索引标签不索引日志内容", "knowledge"),
        ("ClickHouse适合OLAP分析场景", "knowledge"),
        ("Kafka保证分区内消息有序", "knowledge"),
        ("Envoy支持gRPC原生代理", "knowledge"),
        ("ArgoCD实现GitOps自动同步", "knowledge"),
        ("MinIO兼宾S3 API接口", "knowledge"),
        ("Caddy自动获取Let\'s Encrypt证书", "knowledge"),
        ("Traefik支持自动服务发现", "knowledge"),
        ("Celery使用消息中间件作为Broker", "knowledge"),
        ("SQLAlchemy支持多种数据库后端", "knowledge"),
        ("Pytest使用fixture管理测试依赖", "knowledge"),
        ("FastAPI自动生成OpenAPI文档", "knowledge"),
        ("Django ORM使用Active Record模式", "knowledge"),
        ("Next.js支持SSR和SSG两种渲染", "knowledge"),
        ("Tailwind CSS使用Utility-first理念", "knowledge"),
        ("Webpack打包生成bundle文件", "knowledge"),
        ("Vite使用ESM实现快速开发启动", "knowledge"),
        ("Playwright支持多浏览器自动化测试", "knowledge"),
        ("Scrapy使用Spider定义爬取逻辑", "knowledge"),
        ("Pandas的DataFrame类似表格数据结构", "knowledge"),
        ("PyTorch使用动态计算图", "knowledge"),
        ("Socket.IO支持实时双向通信", "knowledge"),
        ("OAuth2有四种授权模式", "knowledge"),
        ("CORS需要服务端设置Access-Control头", "knowledge"),
        ("HTTPS使用TLS加密传输数据", "knowledge"),
        ("WebSocket在单TCP连接上全双工通信", "knowledge"),
        ("RESTful API使用HTTP方法语义", "knowledge"),
        ("OAuth2的Authorization Code模式最安全", "knowledge"),
        ("S3提供高持久性对象存储", "knowledge"),
        ("WAF可以防护SQL注入和XSS攻击", "knowledge"),
    ]

    for content, category in facts:
        mm.store(content=content, category=category, importance=0.8)
    mm._bm25_dirty = True
    mm._rebuild_bm25()

    queries = [
        # Paraphrased
        ("Flask开发REST接口的偏好", "Flask"),
        ("bridge网络模式是什么", "bridge"),
        ("PostgreSQL的JSON支持", "JSON"),
        ("Redis淘汰策略选择", "LRU"),
        ("Nginx worker进程配置", "worker"),
        ("Git rebase使用习惯", "rebase"),
        ("K8s最小调度单位", "Pod"),
        ("JWT令牌有效期", "2小时"),
        ("MySQL事务隔离级别", "REPEATABLE"),
        ("Prometheus采集方式", "Pull"),
        # Partial keyword
        ("Flask", "Flask"),
        ("bridge", "bridge"),
        ("JSON", "JSON"),
        ("LRU", "LRU"),
        ("worker进程数", "worker"),
        ("rebase", "rebase"),
        ("Pod", "Pod"),
        ("JWT", "2小时"),
        ("隔离级别", "REPEATABLE"),
        ("Pull", "Pull"),
        # Cross-lingual
        ("Flask framework REST API", "Flask"),
        ("Docker default network bridge", "bridge"),
        ("PostgreSQL JSON type support", "JSON"),
        ("Redis LRU eviction policy", "LRU"),
        ("Nginx worker process count", "worker"),
        ("Git rebase vs merge", "rebase"),
        ("Kubernetes minimum deploy unit", "Pod"),
        ("JWT token expiration", "2小时"),
        ("MySQL isolation level", "REPEATABLE"),
        ("Prometheus pull metrics", "Pull"),
        # Short
        ("Flask", "Flask"),
        ("Docker网络", "bridge"),
        ("PG JSON", "JSON"),
        ("Redis LRU", "LRU"),
        ("Nginx worker", "worker"),
        ("Git rebase", "rebase"),
        ("K8s Pod", "Pod"),
        ("JWT过期", "2小时"),
        ("MySQL隔离", "REPEATABLE"),
        ("Prometheus Pull", "Pull"),
        # Long descriptive
        ("我想了解Python的Flask框架在开发RESTful API方面的使用经验", "Flask"),
        ("Docker容器默认的bridge网络模式是怎么工作的", "bridge"),
        ("PostgreSQL数据库是否支持直接存储和查询JSON格式的数据", "JSON"),
        ("Redis做缓存的时候使用什么样的内存淘汰策略比较好", "LRU"),
        ("Nginx服务器的worker进程数量通常怎么配置", "worker"),
        ("使用Git进行版本控制时rebase和merge哪种方式更好", "rebase"),
        ("Kubernetes集群中最小的可部署单元是什么", "Pod"),
        ("JWT令牌的过期时间一般设置为多长比较合适", "2小时"),
        ("MySQL数据库默认的事务隔离级别是什么", "REPEATABLE"),
        ("Prometheus监控系统采集数据用的是推模式还是拉模式", "Pull"),
    ]

    for q_text, kw in queries:
        results = mm.retrieve(query=q_text, top_k=3)
        found = any(kw in item.content for item in results)
        T.append((f"q_{len(T):03d}", found, f"q='{q_text[:30]}' kw='{kw}' found={found}"))

    return _run("exp51_adversarial_retrieval", T, 0.80)



# ============================================================
# Exp52: Concurrent Stress (0 error rate)
# ============================================================

def exp52_concurrent_stress():
    """10 threads x 50 ops = 500 total, measure error rate."""
    from src.memory.memory_manager import MemoryManager
    print("\n--- Exp52: Concurrent Stress ---")
    db = _tmp()
    mm = MemoryManager(db_path=db)
    T = []
    errors = []
    lock = threading.Lock()
    op_count = [0]
    queries = ["Python Docker部署", "Redis缓存配置", "Nginx反向代理",
               "PostgreSQL优化", "K8s集群管理", "React前端",
               "JWT认证", "日志方案", "CI/CD流水线", "微服务架构"]

    def worker(tid):
        for i in range(50):
            try:
                if i % 3 == 0:
                    mm.store(content=f"线程{tid}操作{i}: 关于{queries[i % len(queries)]}的记忆",
                             category="interaction", importance=random.uniform(0.3, 0.9))
                else:
                    mm.retrieve(query=queries[(tid + i) % len(queries)], top_k=3)
                with lock:
                    op_count[0] += 1
            except Exception as e:
                with lock:
                    errors.append((tid, i, str(e)))

    threads = []
    t0 = time.perf_counter()
    for tid in range(10):
        t = threading.Thread(target=worker, args=(tid,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    total_ops = op_count[0]
    error_rate = len(errors) / total_ops if total_ops else 1.0
    throughput = total_ops / elapsed if elapsed > 0 else 0

    T.append(("zero_error_rate", error_rate == 0.0,
              f"errors={len(errors)}/{total_ops} rate={error_rate:.4f}"))
    T.append(("throughput", throughput > 10, f"{throughput:.1f} ops/s in {elapsed:.2f}s"))
    T.append(("total_ops", total_ops >= 400, f"expected=500 got={total_ops}"))
    if errors:
        for tid, op, err in errors[:5]:
            T.append(("error_detail", False, f"tid={tid} op={op} err={err[:80]}"))
    try:
        os.unlink(db)
    except OSError:
        pass
    return _run("exp52_concurrent_stress", T, 0.90)


# ============================================================
# Exp53: Noisy Resilience (50+ cases)
# ============================================================

def exp53_noisy_resilience():
    """Test with garbled input, long text, empty, special chars."""
    from src.extraction.zero_llm_pipeline import ZeroLLMPipeline
    from src.memory.memory_manager import MemoryManager
    print("\n--- Exp53: Noisy Resilience ---")
    pipe = ZeroLLMPipeline()
    db = _tmp()
    mm = MemoryManager(db_path=db)
    T = []

    # Empty and whitespace (10)
    for i, text in enumerate(["", " ", "  ", "\n", "\t", "   ", "\r\n", " \t ", "  \n  ", "\n\n"]):
        try:
            result = pipe.classify(text)
            ok = result in ("trivial", "unknown", "")
            T.append((f"empty_{i:02d}", ok, f"input={repr(text)} => {result}"))
        except Exception as e:
            T.append((f"empty_{i:02d}", False, f"exception: {e}"))

    # Store and retrieve empty (5)
    for i in range(5):
        try:
            mm.store(content=f"有效记忆{i}", category="interaction", importance=0.5)
            results = mm.retrieve(query="", top_k=3)
            T.append((f"empty_query_{i}", True, f"returned {len(results)}"))
        except Exception as e:
            T.append((f"empty_query_{i}", False, f"exception: {e}"))

    # Garbled / random (8)
    for i, text in enumerate(["asdkjhqweiuqwhjeasdkjh", "1234567890!@#$%^&*()",
                               "aaaaaaaaaaaaaaaaaaaa", "abc abc abc abc abc",
                               "????????????", "!!!!....,,,,;;;;",
                               "00000000000000", "zzzzzzzzzzzzzzzzzz"]):
        try:
            result = pipe.classify(text)
            T.append((f"garbled_{i:02d}", True, f"'{text[:20]}' => {result}"))
        except Exception as e:
            T.append((f"garbled_{i:02d}", False, f"exception: {e}"))

    # Extremely long text (5)
    for i in range(5):
        long_text = "我喜欢用Python开发项目 " * 200
        try:
            result = pipe.classify(long_text)
            T.append((f"long_{i:02d}", True, f"len={len(long_text)} => {result}"))
        except Exception as e:
            T.append((f"long_{i:02d}", False, f"exception: {e}"))

    # Special characters (8)
    for i, text in enumerate(["我喜欢用C++开发",
                               "URL: https://example.com/api?key=123",
                               "正则: ^[a-z]+\\d{2,4}$",
                               "JSON: {key: value, num: 42}",
                               "SQL: SELECT * FROM users WHERE id > 100",
                               "HTML: <div class='test'>hello</div>",
                               "邮箱: test@example.com",
                               "路径: /home/user/project/src"]):
        try:
            result = pipe.classify(text)
            T.append((f"special_{i:02d}", result != "", f"'{text[:30]}' => {result}"))
        except Exception as e:
            T.append((f"special_{i:02d}", False, f"exception: {e}"))

    # Mixed content (8)
    for i, text in enumerate(["我喜欢用Python开发项目",
                               "Docker部署太方便了",
                               "用Vue开发前端比React简单多了",
                               "Redis缓存真的是性能优化的利器",
                               "K8s集群管理太复杂了",
                               "JWT认证方案比Session更好用",
                               "Nginx配置反向代理超级方便",
                               "PostgreSQL比MySQL功能更强大"]):
        try:
            result = pipe.classify(text)
            T.append((f"mixed_{i:02d}", result != "", f"len={len(text)} => {result}"))
        except Exception as e:
            T.append((f"mixed_{i:02d}", False, f"exception: {e}"))

    # Store noisy data and retrieve (5)
    for i in range(5):
        try:
            mm.store(content=f"测试记忆{i}: 关于Python开发的偏好和经验",
                     category="preference", importance=0.7)
            results = mm.retrieve(query="Python", top_k=3)
            T.append((f"noisy_{i:02d}", len(results) > 0, f"found {len(results)}"))
        except Exception as e:
            T.append((f"noisy_{i:02d}", False, f"exception: {e}"))

    return _run("exp53_noisy_resilience", T, 0.80)



# ============================================================
# Exp54: Cross-Session Persistence (30+ cases)
# ============================================================

def exp54_cross_session():
    """Store with one MemoryManager, retrieve with new instance on same db."""
    from src.memory.memory_manager import MemoryManager
    print("\n--- Exp54: Cross-Session Persistence ---")
    T = []
    db = _tmp()

    facts = [
        ("Python是团队的主要开发语言", "preference", 0.9),
        ("Docker用于容器化部署", "knowledge", 0.8),
        ("Redis作为缓存中间件", "knowledge", 0.8),
        ("Nginx配置反向代理", "knowledge", 0.7),
        ("PostgreSQL存储业务数据", "knowledge", 0.8),
        ("Git管理代码版本", "knowledge", 0.7),
        ("Prometheus监控服务状态", "knowledge", 0.7),
        ("JWT用于身份认证", "knowledge", 0.8),
        ("CI/CD使用GitHub Actions", "preference", 0.7),
        ("日志收集使用ELK Stack", "knowledge", 0.6),
        ("K8s管理容器编排", "knowledge", 0.8),
        ("React开发前端界面", "preference", 0.7),
        ("Flask开发后端API", "preference", 0.7),
        ("Celery处理异步任务", "knowledge", 0.6),
        ("MinIO提供对象存储", "knowledge", 0.6),
    ]

    mm1 = MemoryManager(db_path=db)
    for content, category, importance in facts:
        mm1.store(content=content, category=category, importance=importance)
    T.append(("session1_store", True, f"stored {len(facts)} facts"))

    results1 = mm1.retrieve(query="Python", top_k=3)
    T.append(("session1_retrieve", len(results1) > 0, f"found {len(results1)}"))

    # Session 2: new instance, same db
    mm2 = MemoryManager(db_path=db)
    test_queries = [
        ("Python开发语言", "Python"), ("Docker容器", "Docker"),
        ("Redis缓存", "Redis"), ("Nginx代理", "Nginx"),
        ("PostgreSQL数据库", "PostgreSQL"), ("Git版本控制", "Git"),
        ("Prometheus监控", "Prometheus"), ("JWT认证", "JWT"),
        ("GitHub Actions", "GitHub Actions"), ("ELK日志", "ELK"),
        ("K8s容器编排", "K8s"), ("React前端", "React"),
        ("Flask API", "Flask"), ("Celery任务", "Celery"), ("MinIO存储", "MinIO"),
    ]

    for query_text, keyword in test_queries:
        results = mm2.retrieve(query=query_text, top_k=5)
        found = any(keyword in item.content for item in results)
        T.append(("cross_session", found, f"q='{query_text}' kw='{keyword}'"))

    # Session 3: add more data, verify consistency
    mm3 = MemoryManager(db_path=db)
    mm3.store(content="MongoDB用于文档存储", category="knowledge", importance=0.7)
    results3 = mm3.retrieve(query="MongoDB", top_k=3)
    T.append(("session3_add", any("MongoDB" in r.content for r in results3), "MongoDB found"))
    results3_old = mm3.retrieve(query="Python", top_k=3)
    T.append(("session3_old", any("Python" in r.content for r in results3_old), "Python still accessible"))

    import sqlite3
    conn = sqlite3.connect(db)
    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    conn.close()
    T.append(("total_count", count >= 16, f"total={count}"))
    try:
        os.unlink(db)
    except OSError:
        pass
    return _run("exp54_cross_session", T, 0.90)



# ============================================================
# Exp55: Integration Pipeline (20+ cases)
# ============================================================

def exp55_integration_pipeline():
    """Full pipeline: text -> classify -> store -> conflict_check -> retrieve -> verify."""
    from src.extraction.zero_llm_pipeline import ZeroLLMPipeline
    from src.knowledge.knowledge_engine import KnowledgeEngine
    from src.memory.memory_manager import MemoryManager
    print("\n--- Exp55: Integration Pipeline ---")
    T = []
    pipe = ZeroLLMPipeline()
    db_mem = _tmp()
    db_kg = _tmp()
    mm = MemoryManager(db_path=db_mem)
    ke = KnowledgeEngine(db_path=db_kg)

    pipeline_cases = [
        ("我一直偏好用Python开发后端服务", "preference", "Python后端", "Python"),
        ("Docker容器默认使用bridge网络模式", "knowledge", "Docker网络", "Docker"),
        ("Redis的LRU淘汰策略在缓存场景很实用", "knowledge", "Redis LRU", "Redis"),
        ("今天修复了Nginx配置的SSL证书过期问题", "episode", "Nginx SSL", "Nginx"),
        ("以后新项目都默认用TypeScript加React", "preference", "TypeScript React", "TypeScript"),
        ("PostgreSQL的MVCC机制比MySQL更成熟", "knowledge", "PostgreSQL MVCC", "PostgreSQL"),
        ("昨天部署了Kubernetes集群到生产环境", "episode", "K8s部署", "Kubernetes"),
        ("我们团队习惯用Git rebase保持历史整洁", "preference", "Git rebase", "Git"),
        ("Elasticsearch使用倒排索引来实现全文搜索", "knowledge", "ES搜索原理", "Elasticsearch"),
        ("Prometheus的Pull模式更适合微服务监控", "knowledge", "Prometheus采集", "Prometheus"),
        ("JWT Token过期时间设置为2小时是最佳实践", "knowledge", "JWT过期", "JWT"),
        ("下次迭代要把Falsh换成FastAPI", "preference", "Flask FastAPI", "FastAPI"),
        ("Kafka保证同一个分区内消息是有序的", "knowledge", "Kafka消息顺序", "Kafka"),
        ("完成了从Jenkins到GitHub Actions的CI迁移", "episode", "CI迁移", "GitHub Actions"),
        ("默认用Tailwind CSS写样式比传统CSS快很多", "preference", "Tailwind CSS", "Tailwind"),
        ("gRPC使用Protocol Buffers序列化比JSON效率高", "knowledge", "gRPC序列化", "gRPC"),
        ("建议以后监控都用Prometheus加Grafana方案", "preference", "监控方案", "Prometheus"),
        ("Ansible Playbook使用YAML格式编写自动化脚本", "knowledge", "Ansible脚本", "Ansible"),
        ("上周排查了一个Redis内存泄漏的bug", "episode", "Redis内存泄漏", "Redis"),
        ("Consul的服务发现在微服务架构中必不可少", "knowledge", "Consul服务发现", "Consul"),
    ]

    for i, (text, expected_cat, query, keyword) in enumerate(pipeline_cases):
        cat = pipe.classify(text)
        classify_ok = cat == expected_cat
        mem_item = mm.store(content=text, category=expected_cat, importance=0.8)
        store_ok = mem_item is not None
        ke.store(content=text, category=expected_cat, confidence=0.8)
        conflicts = ke.detect_conflicts_v2(text)
        mm._bm25_dirty = True
        results = mm.retrieve(query=query, top_k=5)
        retrieve_ok = any(keyword in item.content for item in results)
        pipeline_ok = classify_ok and store_ok and retrieve_ok
        T.append((f"pipeline_{i:02d}", pipeline_ok,
                  f"cat={cat}(exp={expected_cat}) store={store_ok} conflicts={len(conflicts)} retrieve={retrieve_ok}"))

    for p in [db_mem, db_kg]:
        try:
            os.unlink(p)
        except OSError:
            pass
    return _run("exp55_integration_pipeline", T, 0.85)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Exp48-55: Adversarial Stress Test v2")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    experiments = [
        exp48_adversarial_preference, exp49_adversarial_conflict,
        exp50_extreme_scale, exp51_adversarial_retrieval,
        exp52_concurrent_stress, exp53_noisy_resilience,
        exp54_cross_session, exp55_integration_pipeline,
    ]

    for fn in experiments:
        try:
            fn()
        except Exception as e:
            name = fn.__name__
            RESULTS[name] = {"error": str(e), "target_met": False}
            print(f"\n!!! {name} CRASHED: {e}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, result in RESULTS.items():
        if "error" in result:
            status = "CRASH"
            detail = result["error"]
        else:
            status = "PASS" if result.get("target_met") else "CHECK"
            detail = f"acc={result.get('accuracy', 'N/A')} target_met={result.get('target_met')}"
        print(f"  {status}  {name}: {detail}")

    total_pass = sum(1 for r in RESULTS.values() if r.get("target_met"))
    total = len(RESULTS)
    print(f"\nTotal: {total_pass}/{total} passed")

    out_dir = Path(__file__).parent.parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"exp48_55_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
