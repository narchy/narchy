![NARchy Logo](https://bytebucket.org/seh/narchy/raw/skynet2/doc/narchy_banner.jpg)
# Metanarchy Repository

This repository hosts a collection of interconnected Java projects centered around artificial intelligence, cognitive computing, and dynamic user interfaces.

## Projects Overview

The three main projects in this repository are:

*   **[jcog](./jcog/README.md)**: A collection of foundational Java utilities and frameworks designed to support the development of cognitive software. It provides a wide array of tools, from specialized data structures and machine learning algorithms to graph manipulation and real-time utilities.
    *   **Potential Applications:** Building blocks for intelligent agents, semantic web technologies, data mining tools, and educational software for AI concepts.
    *   **Future Development:** Expansion of machine learning algorithms, integration with more data sources, and development of more sophisticated graph algorithms.

*   **[narchy](./narchy/README.md)**: An open-source Non-Axiomatic Reasoning System (NARS) derived from OpenNARS. NARchy is designed for AI systems that operate with insufficient knowledge and resources, enabling them to learn from experience and adapt to dynamic environments. It explores advanced temporal reasoning, scalability, and sensorimotor integration.
    *   **Potential Applications:** Autonomous robotics, intelligent control systems, natural language understanding, and AI-driven scientific discovery.
    *   **Future Development:** Enhancements in real-time learning capabilities, improved long-term memory management, and broader integration with sensorimotor systems.

*   **[spacegraph](./spacegraph/README.md)**: A versatile framework for building dynamic, interactive 2D and 3D applications. It combines a declarative UI model with OpenGL acceleration, physics integration (2D/3D), and a rich audio engine, suitable for simulations, data visualizations, games, and complex UIs.
    *   **Potential Applications:** Interactive simulations for training and education, advanced data visualization tools, development of indie games, and user interfaces for complex systems.
    *   **Future Development:** Support for more advanced rendering techniques (e.g., VR/AR), expansion of physics engine capabilities, and a richer set of UI components.

**Interrelations:**
These projects are designed to be complementary, fostering a synergistic environment for AI development:
*   `jcog` serves as a core utility library. It provides essential, high-performance tools and data structures that are leveraged by both `narchy` for its cognitive architecture and `spacegraph` for its rendering and interaction capabilities. For example, `jcog`'s graph algorithms can be used by `narchy` for knowledge representation and by `spacegraph` for scene graph management.
*   `narchy`, with its foundation in `jcog`, can be the "brain" in sophisticated AI applications. It can employ `spacegraph` as a visualization layer to render its internal reasoning processes or to create interactive environments where AI agents, controlled by `narchy`, can perceive and act. This allows for the development of embodied AI systems.
*   `spacegraph` can utilize utilities from `jcog` for tasks like managing complex data structures for its scenes or optimizing rendering algorithms. It can also act as a sophisticated front-end for applications powered by `narchy`, providing intuitive ways to display the outcomes of reasoning processes or to allow human users to interact with `narchy`-driven agents in simulated or real-world scenarios.

Explore each project's respective `README.md` for more detailed information.

# Install

## Prerequisites
*   **JDK 25 (or newer):** This project requires Java Development Kit version 25.
    *   You can download OpenJDK 25 Early Access builds from [jdk.java.net/25](http://jdk.java.net/25/).
    *   Alternatively, look for official JDK 25 releases from Oracle or other vendors like Adoptium as they become available.
*   **Apache Maven:** Ensure Maven is installed to build the project.

## Build & JDK 25 Compatibility Status
This project is configured to compile with JDK 25 and utilizes Java preview features.

**Important JVM Arguments:**
- The `jcog` module requires the following JVM argument for proper operation due to its use of reflection on JDK internal classes:
  `--add-opens java.base/java.io=ALL-UNNAMED`
- If native libraries are used (e.g., by the Prolog engine in `jcog`), the following argument is also necessary:
  `--enable-native-access=ALL-UNNAMED` (This is generally included in the root POM's default `jvmArgs`).

**Current Build Issues (as of [current date/last update]):**
- The `jcog` module, which is a core dependency for other modules like `narchy`, is currently experiencing build timeouts in the automated environment. This prevents its artifacts from being installed locally.
- Consequently, modules depending on `jcog` (e.g., `narchy`) will fail to build due to unresolved dependencies.
- The root cause of the `jcog` build timeout needs further investigation in a more resource-rich or interactive environment.

The Maven build is configured with `<source>25</source>`, `<target>25</target>`, and `<enablePreview>true</enablePreview>` in the main `pom.xml` to support JDK 25.

# Use

To get started with these projects, follow these basic steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/actual-repository-url/metanarchy.git # TODO: Replace with the correct repository URL
    cd metanarchy
    ```
    **Note:** Please replace `https://github.com/actual-repository-url/metanarchy.git` with the actual URL of this repository.

2.  **Build with Maven:**
    The standard way to build all modules in the project is:
    ```bash
    # Clean the project and build all modules
    mvn clean install
    ```

    ### Building Specific Modules
    You can also build specific modules and their dependencies:
    ```bash
    # Build a specific module (e.g., jcog) and its dependencies
    mvn clean install -pl jcog -am
    ```
    ```bash
    # Build only a specific module (e.g., jcog) without its upstream/downstream dependencies
    # This is useful if other modules are already built and up-to-date.
    mvn clean install -pl jcog
    ```

    ### Running Tests
    ```bash
    # Run all tests in the project
    mvn test
    ```
    ```bash
    # Run tests for a specific module (e.g., jcog)
    mvn test -pl jcog
    ```

3.  **IDE Setup:**
    Import the projects into your preferred Java IDE (e.g., IntelliJ IDEA, Eclipse) as Maven projects. The IDE should automatically recognize the project structure and handle dependencies.

Refer to the individual `README.md` files within each project directory for more specific setup instructions, dependencies, and usage examples.

## Recommended VM Arguments
The following are some recommended starting VM arguments. These may need adjustment based on your application's specific needs, your environment, and the garbage collector in use.
```bash
-Xmx2g -da -dsa -XX:+UseNUMA -XX:MaxGCPauseMillis=1
```
*   `-Xmx2g`: Sets the maximum heap size to 2 gigabytes.
*   `-da` / `-dsa`: Disables assertions / Disables system assertions. (Often used as `-ea` to enable assertions during development).
*   `-XX:+UseNUMA`: Enables NUMA (Non-Uniform Memory Access) interleaving, which can improve performance on NUMA hardware.
*   `-XX:MaxGCPauseMillis=1`: Sets a target for maximum GC pause time to 1 millisecond.
    *   **Caution**: This is a very aggressive target. Achieving such low pause times consistently depends heavily on the chosen GC algorithm (e.g., ZGC, Shenandoah), workload, and heap size. If not using a low-pause GC or if encountering performance issues, consider removing this option or adjusting its value to something more conservative (e.g., `100` or `200`).

# Code Quality, Testing, and Build Infrastructure

This project utilizes several Maven plugins to enhance the build process, enforce code quality, and identify potential issues.

*   **`maven-enforcer-plugin`**: This plugin enforces certain rules and constraints during the build. It is configured to ensure dependency convergence, meaning that all transitive dependencies resolve to the same version, preventing potential conflicts and ensuring a stable build.

*   **Code Quality Checks (Linters & Static Analysis):**
    ```bash
    # Run Checkstyle
    mvn checkstyle:check
    ```
    ```bash
    # Run PMD
    mvn pmd:check
    ```
    ```bash
    # Run SpotBugs
    mvn spotbugs:check
    ```
    ```bash
    # Run all site generation phases, which usually include these checks and generates reports
    mvn site
    ```
    (Note: `mvn site` often runs these checks if they are configured in the `reporting` section of the `pom.xml`.)

*   **Mutation Testing (`pitest-maven`)**: This plugin performs mutation testing to evaluate the effectiveness of unit tests. It introduces small changes (mutations) into the codebase and checks if the existing tests can detect these changes.
    *   The configuration in the root `pom.xml` targets classes and tests across `jcog`, `narchy`, and `spacegraph` modules:
        ```xml
        <targetClasses>
            <param>jcog.*</param>
            <param>narchy.*</param>
            <param>spacegraph.*</param>
        </targetClasses>
        <targetTests>
            <param>jcog.*Test</param>
            <param>narchy.*Test</param>
            <param>spacegraph.*Test</param>
        </targetTests>
        ```
    *   To run mutation tests:
        ```bash
        mvn org.pitest:pitest-maven:mutationCoverage
        ```
    *   **Note**: Mutation testing can be very time-consuming, especially for large codebases.

*   **Dependency Vulnerability Check (`dependency-check-maven`)**: This plugin scans project dependencies for known published security vulnerabilities. It is configured to fail the build if any vulnerability with a CVSS score of 0 or higher is detected.
    *   To run a vulnerability scan:
        ```bash
        mvn org.owasp:dependency-check-maven:check
        ```

*   **Reproducible Builds (`maven-jar-plugin`)**: This plugin is configured to create reproducible builds. It ensures that the generated JAR files have consistent content and metadata, regardless of when or where they are built, by setting a fixed output timestamp.

# Maintenance

## Dependency Updates
```./mvnw versions:display-dependency-updates | fgrep '\-\>'```

## Maven Wrapper
To regenerate the Maven Wrapper (mvnw, mvnw.cmd):
```~/mvn/bin/mvn -N io.takari:maven:wrapper -Dmaven=VERSION```
