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
JDK-25+ http://jdk.java.net/25/

# Use
To get started with these projects, follow these basic steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/metanarchy.git
    cd metanarchy
    ```
    (Replace `your-username` with the actual username or organization if different)

2.  **Build with Maven:**
    ```bash
    # Build all modules from the root directory
    mvn clean install
    ```

3.  **IDE Setup:**
    Import the projects into your preferred Java IDE (e.g., IntelliJ IDEA, Eclipse) as Maven projects. The IDE should automatically handle dependencies and project configurations.

Refer to the individual `README.md` files within each project directory for more specific setup instructions, dependencies, and usage examples.

## VM Arguments
```-Xmx2g -da -dsa -XX:+UseNUMA -XX:MaxGCPauseMillis=1```

# Build Infrastructure

This project utilizes several Maven plugins to enhance the build process, enforce code quality, and identify potential issues.

*   **`maven-enforcer-plugin`**: This plugin enforces certain rules and constraints during the build. It is configured to ensure dependency convergence, meaning that all transitive dependencies resolve to the same version, preventing potential conflicts and ensuring a stable build.
*   **`pitest-maven`**: This plugin performs mutation testing to evaluate the effectiveness of unit tests. It introduces small changes (mutations) into the codebase and checks if the existing tests can detect these changes.
    *   **Note**: The plugin is currently configured with placeholder values for `targetClasses` (`com.example.*`) and `targetTests` (`com.example.*Test`). These should be customized to match the actual package structure of your project for effective mutation testing.
    ```
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
    *   To run mutation tests, you can typically execute: `mvn org.pitest:pitest-maven:mutationCoverage`
*   **`dependency-check-maven`**: This plugin scans project dependencies for known published security vulnerabilities. It is configured to fail the build if any vulnerability with a CVSS score of 0 or higher is detected.
    *   To run a vulnerability scan, execute: `mvn org.owasp:dependency-check-maven:check`
*   **`maven-jar-plugin`**: This plugin is configured to create reproducible builds. It ensures that the generated JAR files have consistent content and metadata, regardless of when or where they are built, by setting a fixed output timestamp.

# Maintenance

## Dependency Updates
```./mvnw versions:display-dependency-updates | fgrep '\-\>'```

## Maven Wrapper
To regenerate the Maven Wrapper (mvnw, mvnw.cmd):
```~/mvn/bin/mvn -N io.takari:maven:wrapper -Dmaven=VERSION```
