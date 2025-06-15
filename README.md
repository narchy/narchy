![NARchy Logo](https://bytebucket.org/seh/narchy/raw/skynet2/doc/narchy_banner.jpg)
# Metanarchy Repository

This repository hosts a collection of interconnected Java projects centered around artificial intelligence, cognitive computing, and dynamic user interfaces.

## Projects Overview

The three main projects in this repository are:

*   **[jcog](./jcog/README.md)**: A collection of foundational Java utilities and frameworks designed to support the development of cognitive software. It provides a wide array of tools, from specialized data structures and machine learning algorithms to graph manipulation and real-time utilities.

*   **[narchy](./narchy/README.md)**: An open-source Non-Axiomatic Reasoning System (NARS) derived from OpenNARS. NARchy is designed for AI systems that operate with insufficient knowledge and resources, enabling them to learn from experience and adapt to dynamic environments. It explores advanced temporal reasoning, scalability, and sensorimotor integration.

*   **[spacegraph](./spacegraph/README.md)**: A versatile framework for building dynamic, interactive 2D and 3D applications. It combines a declarative UI model with OpenGL acceleration, physics integration (2D/3D), and a rich audio engine, suitable for simulations, data visualizations, games, and complex UIs.

**Interrelations:**
These projects are designed to be complementary:
*   `jcog` serves as a core utility library, providing essential tools and data structures that are leveraged by both `narchy` and `spacegraph`.
*   `narchy` utilizes `jcog` for its cognitive functions and can employ `spacegraph` as a visualization layer or for creating interactive environments for its AI agents.
*   `spacegraph` can use utilities from `jcog` and can act as a sophisticated front-end for applications powered by `narchy`, displaying reasoning processes or agent interactions.

Explore each project's respective `README.md` for more detailed information.

# Install
JDK-23+ http://jdk.java.net/23/

# Use
TODO

## VM Arguments
```-Xmx2g -da -dsa -XX:+UseNUMA -XX:MaxGCPauseMillis=1```

# Maintenance

## Dependency Updates
```./mvnw versions:display-dependency-updates | fgrep '\-\>'```

## Maven Wrapper
To regenerate the Maven Wrapper (mvnw, mvnw.cmd):
```~/mvn/bin/mvn -N io.takari:maven:wrapper -Dmaven=VERSION```
