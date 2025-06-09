![NARchy Logo](https://bytebucket.org/seh/narchy/raw/skynet2/doc/narchy_banner.jpg)
# NARchy - Axiomatic/Non-Axiomatic Reasoner
## JCog - Cognition Utilities
## SpaceGraph - Fractal User-Interface

# Install
JDK-23+ http://jdk.java.net/23/

# Use
TODO

## VM Arguments
```-Xmx2g -da -dsa -XX:+UseNUMA -XX:MaxGCPauseMillis=1```

# Maintenence

## Dependency Updates
```./mvnw versions:display-dependency-updates | fgrep '\-\>'```

## Maven Wrapper
To regenerate the Maven Wrapper (mvnw, mvnw.cmd):
```~/mvn/bin/mvn -N io.takari:maven:wrapper -Dmaven=VERSION```
