# File: age-website-master/docs/advanced/advanced.md


# Using Cypher in a CTE Expression

There are no restrictions to using Cypher with CTEs ([Common Table Expressions](https://www.postgresql.org/docs/current/queries-with.html)).

Query:


```postgresql
WITH graph_query as (
    SELECT *
        FROM cypher('graph_name', $$
        MATCH (n)
        RETURN n.name, n.age
    $$) as (name agtype, age agtype)
)
SELECT * FROM graph_query;
```


Results:


<table>
  <tr>
   <td><strong>name</strong>
   </td>
   <td><strong>age</strong>
   </td>
  </tr>
  <tr>
   <td>‘Andres’
   </td>
   <td>36
   </td>
  </tr>
  <tr>
   <td>‘Tobias’
   </td>
   <td>25
   </td>
  </tr>
  <tr>
   <td>‘Peter’
   </td>
   <td>35
   </td>
  </tr>
  <tr>
   <td colspan="2" >3 row(s) returned
   </td>
  </tr>
</table>



# Using Cypher in a Join expression

A Cypher query can be part of a `JOIN` clause.


```
Developers Note
Cypher queries using the CREATE, SET, REMOVE clauses cannot be used in sql queries with JOINs, as they affect the Postgres transaction system. One possible solution is to protect the query with CTEs. See the subsection Using CTEs with CREATE, REMOVE, and SET for more details.
```


Query:


```postgresql
SELECT id, 
    graph_query.name = t.name as names_match,
    graph_query.age = t.age as ages_match
FROM schema_name.sql_person AS t
JOIN cypher('graph_name', $$
        MATCH (n:Person)
        RETURN n.name, n.age, id(n)
$$) as graph_query(name agtype, age agtype, id agtype)
ON t.person_id = graph_query.id
```


Results:


<table>
  <tr>
   <td><strong>id</strong>
   </td>
   <td><strong>names_match</strong>
   </td>
   <td><strong>ages_match</strong>
   </td>
  </tr>
  <tr>
   <td>1
   </td>
   <td>True
   </td>
   <td>True
   </td>
  </tr>
  <tr>
   <td>2
   </td>
   <td>False
   </td>
   <td>True
   </td>
  </tr>
  <tr>
   <td>3
   </td>
   <td>True
   </td>
   <td>False
   </td>
  </tr>
  <tr>
   <td colspan="3" >3 row(s) returned
   </td>
  </tr>
</table>


# Cypher in SQL expressions

Cypher cannot be used in an expression— the query must exist in the `FROM` clause of a query. However, if the cypher query is placed in a subquery, it will behave as any SQL style query.


## Using Cypher with '='

When writing a cypher query that is known to return one column and one row, the '=' comparison operator may be used.


```postgresql
SELECT t.name FROM schema_name.sql_person AS t
where t.name = (
    SELECT a
    FROM cypher('graph_name', $$
    	  MATCH (v)
        RETURN v.name
    $$) as (name varchar(50))
    ORDER BY name
    LIMIT 1);
```


Results:


<table>
  <tr>
   <td><strong>name</strong>
   </td>
   <td><strong>age</strong>
   </td>
  </tr>
  <tr>
   <td>‘Andres’
   </td>
   <td>36
   </td>
  </tr>
  <tr>
   <td colspan="2" >3 row(s) returned
   </td>
  </tr>
</table>



## Working with Postgres's IN Clause

When writing a cypher query that is known to return one column, but may have multiple rows. The `IN` operator may be used.

Query:


```postgresql
SELECT t.name, t.age FROM schema_name.sql_person as t 
where t.name in (
    SELECT *
    FROM cypher('graph_name', $$
        MATCH (v:Person)
        RETURN v.name 
    $$) as (a agtype));
```


Results:


<table>
  <tr>
   <td><strong>name</strong>
   </td>
   <td><strong>age</strong>
   </td>
  </tr>
  <tr>
   <td>‘Andres’
   </td>
   <td>36
   </td>
  </tr>
  <tr>
   <td>‘Tobias’
   </td>
   <td>25
   </td>
  </tr>
  <tr>
   <td>‘Peter’
   </td>
   <td>35
   </td>
  </tr>
  <tr>
   <td colspan="2" >3 row(s) returned
   </td>
  </tr>
</table>



## Working with the Postgres EXISTS Clause

When writing a cypher query that may have more than one column and row returned. The `EXISTS` operator may be used.

Query:


```postgresql
SELECT t.name, t.age
FROM schema_name.sql_person as t
WHERE EXISTS (
    SELECT *
    FROM cypher('graph_name', $$
	  MATCH (v:Person)
        RETURN v.name, v.age
    $$) as (name agtype, age agtype)
    WHERE name = t.name AND age = t.age
);
```


Results:


<table>
  <tr>
   <td><strong>name</strong>
   </td>
   <td><strong>age</strong>
   </td>
  </tr>
  <tr>
   <td>‘Andres’
   </td>
   <td>36
   </td>
  </tr>
  <tr>
   <td>‘Tobias’
   </td>
   <td>25
   </td>
  </tr>
  <tr>
   <td colspan="2" >3 row(s) returned
   </td>
  </tr>
</table>



## Querying Multiple Graphs

There is no restriction to the number of graphs an SQL statement can query. Users may query multiple graphs simultaneously.


```postgresql
SELECT graph_1.name, graph_1.age, graph_2.license_number
FROM cypher('graph_1', $$
    MATCH (v:Person)
    RETURN v.name, v.age
$$) as graph_1(col_1 agtype, col_2 agtype, col_3 agtype)
JOIN cypher('graph_2', $$
    MATCH (v:Doctor)
    RETURN v.name, v.license_number
$$) as graph_2(name agtype, license_number agtype)
ON graph_1.name = graph_2.name
```

Results:


<table>
  <tr>
   <td><strong>name</strong>
   </td>
   <td><strong>age</strong>
   </td>
   <td><strong>license_number</strong>
   </td>
  </tr>
  <tr>
   <td>‘Andres’
   </td>
   <td>36
   </td>
   <td>1234567890
   </td>
  </tr>
  <tr>
   <td colspan="3" >3 row(s) returned
   </td>
  </tr>
</table>






# File: age-website-master/docs/advanced/plpgsql.md
# PL/pgSQL Functions

Cypher commands can be run in [PL/pgSQL](https://www.postgresql.org/docs/11/plpgsql-overview.html) functions without restriction.

Data Setup
```postgresql
SELECT *
FROM cypher('imdb', $$
	CREATE (toby:actor {name: 'Toby Maguire'}),
		(tom:actor {name: 'Tom Holland'}),
		(willam:actor {name: 'Willam Dafoe'}),
		(robert:actor {name: 'Robert Downey Jr'}),
		(spiderman:movie {title: 'Spiderman'}),
		(no_way_home:movie {title: 'Spiderman: No Way Home'}),
		(homecoming:movie {title: 'Spiderman: Homecoming'}),
		(ironman:movie {title: 'Ironman'}),
		(tropic_thunder:movie {title: 'Tropic Thunder'}),
		(toby)-[:acted_in {role: 'Peter Parker', alter_ego: 'Spiderman'}]->(spiderman),
		(willam)-[:acted_in {role: 'Norman Osborn', alter_ego: 'Green Goblin'}]->(spiderman),
		(toby)-[:acted_in {role: 'Toby Maguire'}]->(tropic_thunder),
		(robert)-[:acted_in {role: 'Kirk Lazarus'}]->(tropic_thunder),
		(robert)-[:acted_in {role: 'Tony Stark', alter_ego: 'Ironman'}]->(homecoming),
		(tom)-[:acted_in {role: 'Peter Parker', alter_ego: 'Spiderman'}]->(homecoming),
		(tom)-[:acted_in {role: 'Peter Parker', alter_ego: 'Spiderman'}]->(no_way_home),
		(toby)-[:acted_in {role: 'Peter Parker', alter_ego: 'Spiderman'}]->(no_way_home),
		(willam)-[:acted_in {role: 'Norman Osborn', alter_ego: 'Green Goblin'}]->(no_way_home)
$$) AS (a agtype);
```

Function Creation
```postgresql
CREATE OR REPLACE FUNCTION get_all_actor_names()
RETURNS TABLE(actor agtype)
LANGUAGE plpgsql
AS $BODY$
BEGIN
    LOAD 'age';
    SET search_path TO ag_catalog;

    RETURN QUERY 
    SELECT * 
    FROM ag_catalog.cypher('imdb', $$
        MATCH (v:actor)
        RETURN v.name
    $$) AS (a agtype);
END
$BODY$;
```

Query:
```postgresql
SELECT * FROM get_all_actor_names();
```

Results
<table>
  <tr>
   <td><strong>actor</strong>
   </td>
  </tr>
  <tr>
   <td>"Toby Maguire"</td>
  </tr>
  <tr>
   <td>"Tom Holland"</td>
  </tr>
  <tr>
   <td>"Willam Dafoe"</td>
  </tr>
  <tr>
   <td>"Robert Downey Jr"</td>
  </tr>
  <tr>
   <td>4 row(s) returned
   </td>
  </tr>
</table>

```
Developer's Note:

It's recommended that users use the LOAD 'age' command and set the search_path in the function declaration, to ensure the CREATE FUNCTION command works consistently.
```

## Dynamic Cypher


```postgresql
CREATE OR REPLACE FUNCTION get_actors_who_played_role(role agtype)
RETURNS TABLE(actor agtype, movie agtype)
LANGUAGE plpgsql
AS $function$
DECLARE sql VARCHAR;
BEGIN
        load 'age';
        SET search_path TO ag_catalog;

        sql := format('
		SELECT *
		FROM cypher(''imdb'', $$
			MATCH (actor)-[:acted_in {role: %s}]->(movie:movie)
			RETURN actor.name, movie.title
		$$) AS (actor agtype, movie agtype);
	', role);

        RETURN QUERY EXECUTE sql;

END
$function$;
```

```postgresql
SELECT * FROM get_actors_who_played_role('"Peter Parker"');
```


Results
<table>
  <tr>
   <td><strong>actor</strong></td>
   <td><strong>movie</strong></td>
  </tr>
  <tr>
   <td>"Toby Maguire"</td>
   <td>"Spiderman"</td>
  </tr>
  <tr>
   <td>"Toby Maguire"</td>
   <td>"Spiderman: No Way Home"</td>
  </tr>
  <tr>
   <td>"Tom Holland"</td>
   <td>"Spiderman: No Way Home"</td>
  </tr>
  <tr>
   <td>"Tom Holland"</td>
   <td>"Spiderman: Homecoming"</td>
  </tr>
  <tr>
   <td>4 row(s) returned
   </td>
  </tr>
</table>




# File: age-website-master/docs/advanced/prepared_statements.md
# Prepared Statements

Cypher can run a read query within a Prepared Statement. When using parameters with stored procedures, An SQL Parameter must be placed in the cypher function call. See The [AGE Query Format](../intro/cypher.md#the-age-cypher-query-format) for details.

## Cypher Parameter Format

A cypher parameter is in the format of a `'$'` followed by an identifier. Unlike Postgres parameters, Cypher parameters start with a letter, followed by an alphanumeric string of arbitrary length.

Example: `$parameter_name`


## Prepared Statements Preparation

Preparing Prepared Statements in cypher is an extension of Postgres' stored procedure system. Use the `PREPARE` clause to create a query with the Cypher Function call in it. Do not place Postgres style parameters in the cypher query call, instead place Cypher parameters in the query and place a Postgres parameter as the third argument in the Cypher function call.


```postgresql
PREPARE cypher_stored_procedure(agtype) AS
SELECT *
FROM cypher('expr', $$
    MATCH (v:Person) 
    WHERE v.name = $name //Cypher parameter
    RETURN v
$$, $1) //An SQL Parameter must be placed in the cypher function call
AS (v agtype);
```

## Prepared Statements Execution

When executing the prepared statement, place an agtype map with the parameter values where the Postgres Parameter in the Cypher function call is. The value must be an agtype map or an error will be thrown. Exclude the `'$'` for parameter names.


```postgresql
EXECUTE cypher_prepared_statement('{"name": "Tobias"}');
```



# File: age-website-master/docs/advanced/sql_in_cypher.md
# SQL In Cypher

AGE does not support SQL being directly written in Cypher. However with [user defined functions](../functions/user_functions.md) you can write SQL queries and call them in a cypher command.


```
Developer's Note:

Void and Scalar-Value functions only. Set returning functions are not currently supported.
```


## Create Function
```postgresql
CREATE OR REPLACE FUNCTION public.get_event_year(name agtype) RETURNS agtype AS $$
	SELECT year::agtype
	FROM history AS h
	WHERE h.event_name = name::text
	LIMIT 1;
$$ LANGUAGE sql;
```

## Query
```postgresql
SELECT * FROM cypher('graph_name', $$
	MATCH (e:event)
	WHERE e.year < public.get_event_year(e.name)
	RETURN e.name
$$) as (n agtype);

```

Results
<table>
  <tr>
   <td><strong>name</strong>
   </td>
  </tr>
  <tr>
   <td>"Apache Con 2021"
   </td>
  </tr>
  <tr>
   <td colspan="1" >1 row
   </td>
  </tr>
</table>



# File: age-website-master/docs/clauses/create.md
# CREATE

The `CREATE` clause is used to create graph vertices and edges. 


## Terminal CREATE clauses

A `CREATE` clause that is not followed by another clause is called a terminal clause. When a cypher query ends with a terminal clause, no results will be returned from the cypher function call. However, the cypher function call still requires a column list definition. When cypher ends with a terminal node, define a single value in the column list definition: no data will be returned in this variable.

Query


```postgresql
SELECT * 
FROM cypher('graph_name', $$
    CREATE /* Create clause here, no following clause */
$$) as (a agtype);
```



<table>
  <tr>
   <td><strong>a</strong>
   </td>
  </tr>
  <tr>
   <td>0 row(s) returned
   </td>
  </tr>
</table>

## Create single vertex

Creating a single vertex is done by issuing the following query.

Query


```postgresql
SELECT * 
FROM cypher('graph_name', $$
    CREATE (n)
$$) as (v agtype);
```


Nothing is returned from this query.


<table>
  <tr>
   <td><strong>v</strong>
   </td>
  </tr>
  <tr>
   <td>(0 rows)
   </td>
  </tr>
</table>



## Create multiple vertices

Creating multiple vertices is done by separating them with a comma.

Query


```postgresql
SELECT * 
FROM cypher('graph_name', $$
    CREATE (n), (m)
$$) as (v agtype);
```


Result


<table>
  <tr>
   <td><strong>a</strong>
   </td>
  </tr>
  <tr>
   <td>0 row(s) returned
   </td>
  </tr>
</table>



## Create a vertex with a label

To add a label when creating a vertex, use the following syntax:

Query


```postgresql
SELECT * 
FROM cypher('graph_name', $$
    CREATE (:Person)
$$) as (v agtype);
```


Nothing is returned from this query.

Result


<table>
  <tr>
   <td><strong>v</strong>
   </td>
  </tr>
  <tr>
   <td>0 row(s) returned
   </td>
  </tr>
</table>



## Create a vertex with labels and properties

You can create a vertex with labels and properties at the same time.

Query


```postgresql
SELECT * 
FROM cypher('graph_name', $$
    CREATE (:Person {name: 'Andres', title: 'Developer'})
$$) as (n agtype);
```


Nothing is returned from this query.

Result


<table>
  <tr>
   <td><strong>n</strong>
   </td>
  </tr>
  <tr>
   <td>(0 rows)
   </td>
  </tr>
</table>



## Return created node

You can create and return a node within the same query as follows:

Query


```postgresql
SELECT * 
FROM cypher('graph_name', $$
    CREATE (a {name: 'Andres'})
    RETURN a
$$) as (a agtype);
```


The newly-created node is returned.

Result


<table>
  <tr>
   <td><strong>a</strong>
   </td>
  </tr>
  <tr>
   <td>{id: 0; label: ‘’; properties: {name: ‘Andres’}}::vertex
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>

## Create an edge between two nodes

To create an edge between two vertices, we first `MATCH` the two vertices. Once the nodes are matched, we create an edge between them.

Query


```postgresql
SELECT * 
FROM cypher('graph_name', $$
    MATCH (a:Person), (b:Person)
    WHERE a.name = 'Node A' AND b.name = 'Node B'
    CREATE (a)-[e:RELTYPE]->(b)
    RETURN e
$$) as (e agtype);
```


The created edge is returned by the query.

Result


<table>
  <tr>
   <td><strong>e</strong>
   </td>
  </tr>
  <tr>
   <td>{id: 3; startid: 0, endid: 1; label: ‘RELTYPE’; properties: {}}::edge
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



## Create an edge and set properties

Setting properties on edges is done in a similar manner to setting properties when creating vertices. Note that the values can be any expression.

Query


```postgresql
SELECT * 
FROM cypher('graph_name', $$
    MATCH (a:Person), (b:Person)
    WHERE a.name = 'Node A' AND b.name = 'Node B'
    CREATE (a)-[e:RELTYPE {name:a.name + '<->' + b.name}]->(b)
    RETURN e
$$) as (e agtype);
```


The newly-created edge is returned by the example query.

Result


<table>
  <tr>
   <td><strong>e</strong>
   </td>
  </tr>
  <tr>
   <td>{id: 3; startid: 0, endid: 1; label: ‘RELTYPE’; properties: {name: ‘Node A&lt;->Node B’}}::edge
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



## Create a full path
When you use `CREATE` and a pattern, all parts of the patterns that are not already in scope at this time will be created.

Query


```postgresql
SELECT * 
FROM cypher('graph_name', $$
    CREATE p = (andres {name:'Andres'})-[:WORKS_AT]->(neo)<-[:WORKS_AT]-(michael {name:'Michael'})
    RETURN p
$$) as (p agtype);
```


This query creates three nodes and two relationships simultaneously, assigns the pattern to a path variable, and returns said pattern.

Result
<table>
	<tr>
		<td><strong>p</strong></td>
	</tr>
	<tr>
		<td>
			[{id:0; label: ‘’; properties:{name:’Andres’}}::vertex, <br>{id: 3; startid: 0, endid: 1; label: ‘WORKS_AT’; properties: {}}::edge, <br>{id:1; label: ‘’; properties: {}}::vertex,<br>{id: 3; startid: 2, endid: 1; label: ‘WORKS_AT’; properties: {}}::edge,<br>{id:2; label: ‘’; properties: {name:’Michael’}}::vertex]::path
               </td>
	</tr>
	<tr>
		<td>(1 row)
		</td>
	</tr>
</table>




# File: age-website-master/docs/clauses/delete.md
# DELETE

The `DELETE` clause is used to delete graph elements—nodes, relationships orpaths.

## Terminal DELETE clauses

A `DELETE` clause that is not followed by another clause is called a terminal clause. When a cypher query ends with a terminal clause, no results will be returned from the cypher function call. However, the cypher function call still requires a column list definition. When cypher ends with a terminal node, define a single value in the column list definition: no data will be returned in this variable.


## Introduction

For removing properties, see `REMOVE`.

You cannot delete a node without also deleting edges that start or end on said vertex. Either explicitly delete the vertices,or use `DETACH DELETE`.


## Delete isolated vertices

To delete a vertex, use the `DELETE` clause.

Query


```postgresql
SELECT * 
FROM cypher('graph_name', $$
	MATCH (v:Useless)
	DELETE v
$$) as (v agtype);
```


This will delete the vertices (with label Useless) that have no edges. Nothing is returned from this query.


<table>
  <tr>
   <td><strong>v</strong>
   </td>
  </tr>
  <tr>
   <td>(0 rows)
   </td>
  </tr>
</table>

## Delete all vertices and edges associated with them

Running a `MATCH` clause will collect all nodes— use the `DETACH` option to first delete a vertice's edges, then delete the vertex itself.

Query


```postgresql
SELECT * 
FROM cypher('graph_name', $$
	MATCH (v:Useless)
	DETACH DELETE v
$$) as (v agtype);
```


Nothing is returned from this query.


<table>
  <tr>
   <td><strong>v</strong>
   </td>
  </tr>
  <tr>
   <td>(0 rows)
   </td>
  </tr>
</table>

## Delete edges only

To delete an edge, use the `MATCH` clause to find your edges, then add the variable to the `DELETE` clause.

Query
```postgresql
SELECT * 
FROM cypher('graph_name', $$
	MATCH (n {name: 'Andres'})-[r:KNOWS]->()
	DELETE r
$$) as (v agtype);
```


Nothing is returned from this query.


<table>
  <tr>
   <td><strong>v</strong>
   </td>
  </tr>
  <tr>
   <td>(0 rows)
   </td>
  </tr>
</table>

## Return a deleted vertex

You can return vertices that have been deleted with a `RETURN` clause.

Query
```postgresql
SELECT *
FROM cypher('graph_name', $$
	MATCH (n {name: 'A'})
	DELETE n
	RETURN n
$$) as (a agtype);

```

<table>
  <tr>
   <td><strong>v</strong>
   </td>
  </tr>
  <tr><td>{"id": 281474976710659, "label": "", "properties": {"name": "A"}}::vertex</td></tr>
  <tr>
   <td>(1 rows)
   </td>
  </tr>
</table>





# File: age-website-master/docs/clauses/limit.md
# LIMIT

`LIMIT` constrains the number of records in the output.

## Introduction

`LIMIT` accepts any expression that evaluates to a positive integer.


## Return a subset of the rows

To return a subset of the result, starting from the top, use the following syntax:

Query


```postgresql
SELECT * 
FROM cypher('graph_name', $$
	MATCH (n) RETURN n.name
	ORDER BY n.name
	LIMIT 3
$$) as (names agtype);
```


The name property of the matched node `n` is returned, with a limit of 3.

Result


<table>
  <thead>
   <td><strong>names</strong>
   </td>
  <thead>
  <tr>
   <td>"A"
   </td>
  </tr>
  <tr>
   <td>"B"
   </td>
  </tr>
  <tr>
   <td>"C"
   </td>
  </tr>
  <tr>
   <td>3 rows
   </td>
  </tr>
</table>

## Using an expression with LIMIT to return a subset of the rows

`LIMIT` accepts any expression that evaluates to a positive integer as long as it is not referring to any external variables:

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
	MATCH (n)
	RETURN n.name
	ORDER BY n.name
	LIMIT toInteger(3 * rand()) + 1
$$) as (names agtype);

```

Returns one to three top items.

Result


<table>
  <thead>
   <td><strong>names</strong>
   </td>
  <thead>
  <tr>
   <td>"A"
   </td>
  </tr>
  <tr>
   <td>"B"
   </td>
  </tr>
  <tr>
   <td>2 rows
   </td>
  </tr>
</table>


# File: age-website-master/docs/clauses/match.md
# MATCH

The `MATCH` clause allows you to specify the patterns a query will search for in the database. This is the primary way of retrieving data for use in a query.

 A `WHERE` clause often follows a `MATCH` clause to add user-defined restrictions to the matched patterns to manipulate the set of data returned. The predicates are part of the pattern description, and should not be considered a filter applied only after the matching is done. This means that `WHERE` should always be put together with the `MATCH` clause it belongs to.

MATCH can occur at the beginning of the query or later, possibly after a `WITH`. If it is the first clause, nothing will have been bound yet, and Cypher will design a search to find the results matching the clause and any associated predicates specified in any `WHERE` clause. Vertices and edges found by this search are available as bound pattern elements, and can be used for pattern matching of sub-graphs. They can also be used in any future clauses, where Cypher will use the known elements, and from there find further unknown elements.

Cypher is a declarative language, and so typically the query itself does not specify the algorithm to use to perform the search. Predicates in `WHERE` parts can be evaluated before pattern matching, during pattern matching, or after the match is found.


## Basic vertex finding


### Get all Vertices

By specifying a pattern with a single vertex and no labels, all vertices in the graph will be returned.

Query

```postgresql
SELECT * FROM cypher('graph_name', $$
MATCH (v)
RETURN v
$$) as (v agtype);
```


Returns all vertices in the database.


<table>
  <tr>
   <td><strong>v</strong>
   </td>
  </tr>
  <tr>
   <td>{id: 0; label: ‘Person’; properties: {name: ‘Charlie Sheen’}}::vertex
   </td>
  </tr>
  <tr>
   <td>{id: 1; label: ‘Person’; properties: {name: ‘Martin Sheen’}}::vertex
   </td>
  </tr>
  <tr>
   <td>{id: 2; label: ‘Person’; properties: {name: ‘Michael  Douglas’}}::vertex
   </td>
  </tr>
  <tr>
   <td>{id: 3; label: ‘Person’; properties: {name: ‘Oliver Stone’}}::vertex
   </td>
  </tr>
  <tr>
   <td>{id: 4; label: ‘Person’; properties: {name: ‘Rob Reiner’}}::vertex
   </td>
  </tr>
  <tr>
   <td>{id: 5; label: ‘Movie’; properties: {name: ‘Wall Street’}}::vertex
   </td>
  </tr>
  <tr>
   <td>{id: 6; label: ‘Movie’; properties: {title: ‘The American President’}}::vertex
   </td>
  </tr>
  <tr>
   <td>7 row(s) returned
   </td>
  </tr>
</table>



### Get all vertices with a label

Getting all vertices with a label is done with a single node pattern where the vertex has the label specified as follows:

Query


```postgresql
SELECT * FROM cypher('graph_name', $$
MATCH (movie:Movie)
RETURN movie.title
$$) as (title agtype);
```


Returns all the movies in the database.


<table>
  <tr>
   <td><strong>title</strong>
   </td>
  </tr>
  <tr>
   <td>‘Wall Street’
   </td>
  </tr>
  <tr>
   <td>‘The American President’
   </td>
  </tr>
  <tr>
   <td>2 row(s) returned
   </td>
  </tr>
</table>



### Related Vertices

The symbol `-[]-` specifies an edge, without specifying the type or direction of the edge.

Query


```postgresql
SELECT * FROM cypher('graph_name', $$
MATCH (director {name: 'Oliver Stone'})-[]-(movie)
RETURN movie.title
$$) as (title agtype);
```


Returns all the movies directed by 'Oliver Stone'.


<table>
  <tr>
   <td><strong>title</strong>
   </td>
  </tr>
  <tr>
   <td>‘Wall Street’
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



### Match with labels

To constrain your pattern with labels on vertices, add it to the vertex in the pattern, using the label syntax.

Query


```postgresql
SELECT * FROM cypher('graph_name', $$
MATCH (:Person {name: 'Oliver Stone'})-[]-(movie:Movie)
RETURN movie.title
$$) as (title agtype);
```


Returns any vertices connected with the `Person` 'Oliver' that are labeled `Movie`.


<table>
  <tr>
   <td><strong>title</strong>
   </td>
  </tr>
  <tr>
   <td>‘Wall Street’
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## Edge basics


### Outgoing Edges

To return directed edges, you may use `->` or `<-` to specify the direction of which the edge points.

Query


```postgresql
SELECT * FROM cypher('graph_name', $$
MATCH (:Person {name: 'Oliver Stone'})-[]->(movie)
RETURN movie.title
$$) as (title agtype);
```


Returns any vertices connected with the `Person` 'Oliver' by an outgoing edge.


<table>
  <tr>
   <td><strong>title</strong>
   </td>
  </tr>
  <tr>
   <td>‘Wall Street’
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



### Directed Edges and variable
 
If a variable is required, either for filtering on properties of the edge, or to return the edge, specify the variable within the edge or vertex you wish to use.

Query


```postgresql
SELECT * FROM cypher('graph_name', $$
MATCH (:Person {name: 'Oliver Stone'})-[r]->(movie)
RETURN type(r)
$$) as (title agtype);
```


Returns the type of each outgoing edge from 'Oliver'.


<table>
  <tr>
   <td><strong>title</strong>
   </td>
  </tr>
  <tr>
   <td>‘DIRECTED’
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



### Match on edge label

When you know the edge label you want to match on, you can specify it by using a colon together with the edge label.

Query


```postgresql
SELECT * FROM cypher('graph_name', $$
MATCH (:Movie {title: 'Wall Street'})<-[:ACTED_IN]-(actor)
RETURN actor.name
$$) as (actors_name agtype);
```


Returns all actors that `ACTED_IN` 'Wall Street'.


<table>
  <tr>
   <td><strong>actors_name</strong>
   </td>
  </tr>
  <tr>
   <td>‘Charlie Sheen’
   </td>
  </tr>
  <tr>
   <td>‘Martin Sheen’
   </td>
  </tr>
  <tr>
   <td>‘Michael  Douglas’
   </td>
  </tr>
  <tr>
   <td>3 row(s) returned
   </td>
  </tr>
</table>



### Match on edge label with a variable

If you want to use a variable to hold the edge, and specify the edge label you want, you can do so by specifying them both.

Query


```postgresql
SELECT * FROM cypher('graph_name', $$
MATCH ({title: 'Wall Street'})<-[r:ACTED_IN]-(actor)
RETURN r.role
$$) as (role agtype);
```


Returns `ACTED_IN` roles for 'Wall Street'.


<table>
  <tr>
   <td><strong>role</strong>
   </td>
  </tr>
  <tr>
   <td>‘Gordon Gekko’
   </td>
  </tr>
  <tr>
   <td>‘Carl Fox’
   </td>
  </tr>
  <tr>
   <td>‘Bud Fox’
   </td>
  </tr>
  <tr>
   <td>3 row(s) returned
   </td>
  </tr>
</table>



### Multiple Edges

Edges can be strung together to match an infinite number of edges. As long as the base pattern `()-[]-()` is followed, users can chain together edges and vertices to match specific patterns.

Query


```postgresql
SELECT * FROM cypher('graph_name', $$
    MATCH (charlie {name: 'Charlie Sheen'})-[:ACTED_IN]->(movie)<-[:DIRECTED]-(director)
    RETURN movie.title, director.name
$$) as (title agtype, name agtype);
```


Returns the movie 'Charlie Sheen' acted in and its director.


<table>
  <tr>
   <td><strong>title</strong>
   </td>
   <td><strong>name</strong>
   </td>
  </tr>
  <tr>
   <td>‘Wall Street’
   </td>
   <td>‘Oliver Stone’
   </td>
  </tr>
  <tr>
   <td colspan="2" >1 row(s) returned
   </td>
  </tr>
</table>


## Variable Length Edges

When the connection between two vertices is of variable length, the list of edges that form the connection can be returned using the following connection.

### Introduction

Rather than describing a long path using a sequence of many vertex and edge descriptions in a pattern, many edges (and the intermediate vertices) can be described by specifying a length in the edge description of a pattern.

```
(u)-[*2]->(v)
```

Which describes a right directed path of three vertices and two edges can be rewritten to:

```
(u)-[]->()-[]->(v)
```

A range length can also be given:


```
(u)-[*3..5]->(v)
```

Which is equivalent to:

```
(u)-[]->()-[]->()-[]->(v) and
(u)-[]->()-[]->()-[]->()-[]->(v) and
(u)-[]->()-[]->()-[]->()-[]->()-[]->(v)
```

The previous example provided gave the edge both an lower and upper bound for the number of edges (and vertices) between `u` and `v`. Either one or both of these binding values can be excluded.


```
(u)-[*3..]->(v)
```

Returns all paths between `u` and `v` that have three or more edges included.

```
(u)-[*..5]->(v)
```

Returns all paths between `u` and `v` that have 5 or fewer edges included.

```
(u)-[*]->(v)
```

Returns all paths between `u` and `v`.


### Example


Query


```postgresql
SELECT * FROM cypher('graph_name', $$
    MATCH p = (actor {name: 'Willam Dafoe'})-[:ACTED_IN*2]-(co_actor)
    RETURN relationships(p)
$$) as (r agtype);
```


Returns the list of edges, including the one that Willam Dafoe acted in and the two Spiderman actors he worked with.


<table>
  <tr>
   <td><strong>r</strong>
   </td>
  </tr>
  <tr>
   <td>[{id: 0; label:"ACTED_IN"; properties: {role: "Green Goblin"}}::edge, {id: 1; label: "ACTED_IN; properties: {role: "Spiderman", actor: "Toby Maguire}}::edge]
   </td>
  </tr>
  <tr>
   <td>[{id: 0; label:"ACTED_IN"; properties: {role: "Green Goblin"}}::edge, {id: 2; label: "ACTED_IN; properties: {role: "Spiderman", actor: "Andrew Garfield"}}::edge]
   </td>
   </td>
  </tr>
  <tr>
   <td colspan="2" >2 row(s) returned
   </td>
  </tr>
</table>



# File: age-website-master/docs/clauses/merge.md
# MERGE

The `MERGE` clause ensures that a pattern exists in the graph. Either the pattern already exists, or it needs to be created.


`MERGE` either matches existing nodes, or creates new data. It’s a combination of `MATCH` and `CREATE`.

For example, you can specify that the graph must contain a node for a user with a certain name. If there isn’t a node with the correct name, a new node will be created and its name property set. When using `MERGE` on full patterns, the behavior is that either the whole pattern matches, or the whole pattern is created. `MERGE` will not partially use existing patterns. If partial matches are needed, this can be accomplished by splitting a pattern up into multiple `MERGE` clauses.

As with `MATCH`, `MERGE` can match multiple occurrences of a pattern. If there are multiple matches, they will all be passed on to later stages of the query.

## Data Setup

```postgresql
SELECT * from cypher('graph_name', $$
CREATE (A:Person {name: "Charlie Sheen", bornIn: "New York"}),
    (B:Person {name: "Michael Douglas", bornIn: "New Jersey"}),
    (C:Person {name: "Rob Reiner", bornIn: "New York"}),
    (D:Person {name: "Oliver Stone", bornIn: "New York"}),
    (E:Person {name: "Martin Sheen", bornIn: "Ohio"})
$$) as (result agtype);
```

## Merge Nodes

### Merge a Node with a Label

By just specifying a pattern with a single vertex and no labels, all vertices in the graph will be returned.

Query

```postgresql
SELECT * FROM cypher('graph_name', $$
MERGE (v:Critic)
RETURN v
$$) as (v agtype);
```

If there exists a vertex with the label 'Critic', that vertex will be returned. Otherwise, the vertex will be created and returned.

<table>
  <tr>
   <td><strong>v</strong>
   </td>
  </tr>
  <tr>
   <td>{id: 0; label: ‘Critic’: properties:{}}::vertex
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>


### Merge Single Vertex with Properties

Merging a vertex node with properties where not all properties match any existing vertex.

Query

```postgresql
SELECT * FROM cypher('graph_name', $$
MERGE (charlie {name: 'Charlie Sheen', age: 10})
RETURN charlie
$$) as (v agtype);
```

If there exists a vertex with the label 'Critic', that vertex will be returned. Otherwise, the vertex will be created and returned.

<table>
  <tr>
   <td><strong>v</strong>
   </td>
  </tr>
  <tr>
   <td>{id: 0; label: ‘Actor’: properties:{name: 'Charlie Sheen', age: 10}}::vertex
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>

If there exists a vertex with all properties, that vertex will be returned. Otherwise, a new vertex with the name 'Charlie Sheen' will be created and returned.


### Merge a Single Vertex Specifying Both Label and Property

Merging a vertex where both label and property constraints match an existing vertex.

Query

```postgresql
SELECT * FROM cypher('graph_name', $$
MERGE (michael:Person {name: 'Michael Douglas'})
RETURN michael.name, michael.bornIn
$$) as (Name agtype, BornIn agtype);
```

'Michael Douglas' will match the existing vertex and the vertex's `name` and `bornIn` properties will be returned.

<table>
  <tr>
   <td><strong>Name</strong></td>
   <td><strong>BornIn</strong></td>
  </tr>
  <tr>
   <td>"Michael Douglas"</td>
   <td>"New Jersey"</td>
  </tr>
  <tr>
   <td>1 row(s) returned</td>
  </tr>
</table>


# File: age-website-master/docs/clauses/order_by.md
# ORDER BY

`ORDER BY` is a sub-clause following `WITH`. ORDER BY specifies that the output should be sorted and how it will be sorted. 

## Introduction

Note that you cannot sort on nodes or relationships, sorting must be done on properties. `ORDER BY` relies on comparisons to sort the output. See Ordering and comparison of values.

In terms of scope of variables, `ORDER BY` follows special rules, depending on if the projecting `RETURN` or `WITH` clause is either aggregating or `DISTINCT`. If it is an aggregating or `DISTINCT` projection, only the variables available in the projection are available. If the projection does not alter the output cardinality (which aggregation and `DISTINCT` do), variables available from before the projecting clause are also available. When the projection clause shadows already existing variables, only the new variables are available.

Lastly, it is not allowed to use aggregating expressions in the `ORDER BY` sub-clause if they are not also listed in the projecting clause. This last rule is to make sure that `ORDER BY` does not change the results, only the order of them.


## Order nodes by property

`ORDER BY` is used to sort the output.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (n)
    WITH n.name as name, n.age as age
    ORDER BY n.name
    RETURN name, age
$$) as (name agtype, age agtype);
```


The nodes are returned, sorted by their name.

Result


<table>
  <tr>
   <td><strong>name</strong>
   </td>
   <td><strong>age</strong>
   </td>
  </tr>
  <tr>
   <td>"A"
   </td>
   <td>34
   </td>
  </tr>
  <tr>
   <td>"B"
   </td>
   <td>34
   </td>
  </tr>
  <tr>
   <td>"C"
   </td>
   <td>32
   </td>
  </tr>
  <tr>
   <td colspan="2" >(1 row)
   </td>
  </tr>
</table>



## Order nodes by multiple properties

You can order by multiple properties by stating each variable in the `ORDER BY` clause. Cypher will sort the result by the first variable listed, and for equal values, go to the next property in the `ORDER BY` clause, and so on.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (n)
    WITH n.name as name, n.age as age
    ORDER BY n.age, n.name
    RETURN name, age
$$) as (name agtype, age agtype);
```


This returns the nodes, sorted first by their age, and then by their name.

Result


<table>
  <tr>
   <td><strong>name</strong>
   </td>
   <td><strong>age</strong>
   </td>
  </tr>
  <tr>
   <td>"C"
   </td>
   <td>32
   </td>
  </tr>
  <tr>
   <td>"A"
   </td>
   <td>34
   </td>
  </tr>
  <tr>
   <td>"B"
   </td>
   <td>34
   </td>
  </tr>
  <tr>
   <td colspan="2" >(1 row)
   </td>
  </tr>
</table>



## Order nodes in descending order

By adding `DESC[ENDING]` after the variable to sort on, the sort will be done in reverse order.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (n)
    WITH n.name AS name, n.age AS age
    ORDER BY n.name DESC
    RETURN name, age
$$) as (name agtype, age agtype);
```


The example returns the nodes, sorted by their name in reverse order.

Result


<table>
  <tr>
   <td><strong>name</strong>
   </td>
   <td><strong>age</strong>
   </td>
  </tr>
  <tr>
   <td>"C"
   </td>
   <td>32
   </td>
  </tr>
  <tr>
   <td>"B"
   </td>
   <td>34
   </td>
  </tr>
  <tr>
   <td>"A"
   </td>
   <td>34
   </td>
  </tr>
  <tr>
   <td colspan="2" >(3 rows)
   </td>
  </tr>
</table>



## Ordering null

When sorting the result set, `null` will always come at the end of the result set for ascending sorting, and first for descending sorting.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (n)
    WITH n.name AS name, n.age AS age, n.height
    ORDER BY n.height
    RETURN name, age, height
$$) as (name agtype, age agtype, height agtype);
```


The nodes are returned sorted by the length property, with a node without that property last. 

Result


<table>
  <tr>
   <td><strong>name</strong>
   </td>
   <td><strong>age</strong>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>"A"
   </td>
   <td>34
   </td>
   <td>170
   </td>
  </tr>
  <tr>
   <td>"C"
   </td>
   <td>32
   </td>
   <td>185
   </td>
  </tr>
  <tr>
   <td>"B"
   </td>
   <td>34
   </td>
   <td>&lt;NULL>
   </td>
  </tr>
  <tr>
   <td colspan="3" >(3 rows)
   </td>
  </tr>
</table>



# File: age-website-master/docs/clauses/remove.md
# REMOVE

The `REMOVE` clause is used to remove properties from vertex and edges.


## Terminal REMOVE clauses

A `REMOVE` clause that is not followed by another clause is a terminal clause. When a cypher query ends with a terminal clause, no results will be returned from the cypher function call. However, the cypher function call still requires a column list definition. When cypher ends with a terminal node, define a single value in the column list definition: no data will be returned in this variable.


## Remove a property

Cypher does not allow storing `null` in properties. Instead, if no value exists, the property is just not there. So, removing a property value on a node or a relationship is also done with `REMOVE`.

Query


```postgresql
SELECT * 
FROM cypher('graph_name', $$
    MATCH (andres {name: 'Andres'})
    REMOVE andres.age
    RETURN andres
$$) as (andres agtype);
```


The node is returned, and no property age exists on it.

Result


<table>
  <tr>
   <td><strong>andres</strong>
   </td>
  </tr>
  <tr>
   <td>{id: 3; label: ‘Person’; properties: {name:"Andres"}}::vertex
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>


# File: age-website-master/docs/clauses/return.md
# RETURN  

In the `RETURN` part of your query, you define which parts of the pattern you want to output. Output can include agtype values, nodes, relationships, or properties.


## Return nodes

To return a node, list it in the `RETURN` statement.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (n {name: 'B'})
    RETURN n
$$) as (n agtype);
```


The example will return the node.

Result


<table>
  <tr>
   <td><strong>n</strong>
   </td>
  </tr>
  <tr>
   <td>{id: 0; label: ‘’ properties: {name: ‘B’}}::vertex
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



## Return edges

To return `n`'s edges, just include it in the `RETURN` list.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (n)-[r:KNOWS]->()
    WHERE n.name = 'A'
    RETURN r
$$) as (r agtype);
```


The relationship is returned by the example.


<table>
  <tr>
   <td><strong>r</strong>
   </td>
  </tr>
  <tr>
   <td>{id: 2; startid: 0; endid: 1; label: ‘KNOWS’ properties: {}}::edge
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



## Return property

To return a property, use the dot separator, as follows:

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (n {name: 'A'})
    RETURN n.name
$$) as (name agtype);
```


The value of the property name gets returned.

Result


<table>
  <tr>
   <td><strong>name</strong>
   </td>
  </tr>
  <tr>
   <td>‘A’
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>


## Return all elements

When you want to return all vertices, edges and paths found in a query, you can use the `*` symbol.

Query

```postgresql
SELECT *
FROM cypher('graph_name', $$
	MATCH (a {name: 'A'})-[r]->(b)
	RETURN *
$$) as (a agtype, b agtype, r agtype);
```


This returns the two vertices, and the edge used in the query.

Result
<table>
  <thead>
  <tr>
   <td><strong>a</strong></td>
   <td><strong>b</strong></td>
   <td><strong>r</strong></td>
  </tr>
  </thead>
  <tbody>
  <tr>
   <td>{"id": 281474976710659, "label": "", "properties": {"age": 55, "name": "A", "happy": "Yes!"}}::vertex 
   </td>
   <td>
{"id": 1125899906842625, "label": "BLOCKS", "end_id": 281474976710660, "start_id": 281474976710659, "properties": {}}::edge
   </td>
   <td>
{"id": 281474976710660, "label": "", "properties": {"name": "B"}}::vertex
   </td>
  </tr>
  <tr>
   <td>{"id": 281474976710659, "label": "", "properties": {"age": 55, "name": "A", "happy": "Yes!"}}::vertex 
   </td>
   <td>
{"id": 1407374883553281, "label": "KNOWS", "end_id": 281474976710660, "start_id": 281474976710659, "properties": {}}::edge
   </td>
   <td>
{"id": 281474976710660, "label": "", "properties": {"name": "B"}}::vertex
   </td>
  </tr>
  <tbody>
   <td colspan="3">(2 rows)
   </td>
  </tr>
</table>

## Variable with uncommon characters

To introduce a placeholder that is made up of characters that are not contained in the English alphabet, you can use the ` to enclose the variable, like this:

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (`This isn\'t a common variable`)
    WHERE `This isn\'t a common variable`.name = 'A'
    RETURN `This isn\'t a common variable`.happy
$$) as (happy agtype);
```


The node with name "A" is returned.

Result


<table>
  <tr>
   <td><strong>happy</strong>
   </td>
  </tr>
  <tr>
   <td>"Yes!"
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



## Aliasing a field

If the name of the field should be different from the expression used, you can rename it by changing the name in the column list definition.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (n {name: 'A'})
    RETURN n.name
$$) as (objects_name agtype);
```


Returns the age property of a node, but renames the field.

Result


<table>
  <tr>
   <td><strong>objects_name</strong>
   </td>
  </tr>
  <tr>
   <td>‘A’
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



## Optional properties

If a property might or might not be there, it will be treated as null if it is missing.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (n)
    RETURN n.age
$$) as (age agtype);
```


This query returns the property if it exists, or null if the property does not exist.

Result


<table>
  <tr>
   <td><strong>age</strong>
   </td>
  </tr>
  <tr>
   <td>55
   </td>
  </tr>
  <tr>
   <td>NULL
   </td>
  </tr>
  <tr>
   <td>(2 rows)
   </td>
  </tr>
</table>



## Other expressions

Any expression can be used as a return item—literals, predicates, properties, functions, and everything else.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (a)
    RETURN a.age > 30, 'I'm a literal', id(a)
$$) as (older_than_30 agtype, literal agtype, id agtype);
```


Returns a predicate, a literal and function call with a pattern expression parameter.

Result


<table>
  <tr>
   <td><strong>older_than_30</strong>
   </td>
   <td><strong>literal</strong>
   </td>
   <td><strong>id</strong>
   </td>
  </tr>
  <tr>
   <td>true
   </td>
   <td>‘I’m a literal’
   </td>
   <td>1
   </td>
  </tr>
  <tr>
   <td colspan="3" >(1 row)
   </td>
  </tr>
</table>



## Unique results

`DISTINCT` retrieves only unique records depending on the fields that have been selected to output.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
MATCH (a {name: 'A'})-[]->(b)
RETURN DISTINCT b
$$) as (b agtype);
```


The node named "B" is returned by the query, but only once.

Result


<table>
  <tr>
   <td><strong>b</strong>
   </td>
  </tr>
  <tr>
   <td>{id: 1; label: ‘’ properties: {name: ‘B’}}::vertex
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



# File: age-website-master/docs/clauses/set.md
# SET

The `SET` clause is used to update labels and properties on vertices and edges


## Terminal SET clauses

A `SET` clause that is not followed by another clause is a terminal clause. When a cypher query ends with a terminal clause, no results will be returned from the cypher function call. However, the cypher function call still requires a column list definition. When cypher ends with a terminal node, define a single value in the column list definition: no data will be returned in this variable.


## Set a property

To set a property on a node or relationship, use `SET`.

Query


```postgresql
SELECT * 
FROM cypher('graph_name', $$
   MATCH (v {name: 'Andres'})
   SET v.surname = 'Taylor'
$$) as (v agtype);
```


The newly changed node is returned by the query.

Result


<table>
  <tr>
   <td><strong>v</strong>
   </td>
  </tr>
  <tr>
   <td>(0 rows)
   </td>
  </tr>
</table>



## Return created vertex

Creating a single vertex is done with the following query:

Query


```postgresql
SELECT * 
FROM cypher('graph_name', $$
    MATCH (v {name: 'Andres'})
    SET v.surname = 'Taylor'
    RETURN v
$$) as (v agtype);
```


The newly changed vertex is returned by the query.

Result


<table>
  <tr>
   <td><strong>v</strong>
   </td>
  </tr>
  <tr>
   <td>{id: 3; label: ‘Person’; properties: {surname:"Taylor", name:"Andres", age:36, hungry:true}}::vertex
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



## Remove a property

Normally a property can be removed by using `REMOVE`, but users can also remove properties using the `SET` command. One example is if the property comes from a parameter.

Query


```postgresql
SELECT * 
FROM cypher('graph_name', $$
    MATCH (v {name: 'Andres'})
    SET v.name = NULL
    RETURN v
$$) as (v agtype);
```


The node is returned by the query, and the name property is now missing.

Result


<table>
  <tr>
   <td><strong>v</strong>
   </td>
  </tr>
  <tr>
   <td>{id: 3; label: ‘Person’; properties: {surname:"Taylor", age:36, hungry:true}}::vertex
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>


## Set multiple properties using one SET clause

If you want to set multiple properties in one query, you can separate them with a comma.

Query


```postgresql
SELECT * 
FROM cypher('graph_name', $$
MATCH (v {name: 'Andres'})
SET v.position = 'Developer', v.surname = 'Taylor'
RETURN v
$$) as (v agtype);
```


Result

<table>
  <tr>
   <td><strong>v</strong>
   </td>
  </tr>
  <tr>
   <td> {"id": 281474976710661, "label": "", "properties": {"name": "Andres", "surname": "Taylor", "position": "Developer"}}:
:vertex
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>





# File: age-website-master/docs/clauses/skip.md
# SKIP

`SKIP` defines from which record to start including the records in the output.

## Introduction

By using `SKIP`, the result set will get trimmed from the top. Please note that no guarantees are made on the order of the returned results unless specified by the `ORDER BY` clause. `SKIP` accepts any expression that evaluates to a positive  integer.

## Skip first three rows

To return a subset of the result, starting from the top, use this syntax:

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
	MATCH (n)
	RETURN n.name
	ORDER BY n.name
	SKIP 3
$$) as (names agtype);
```


The node is returned, and no property age exists on it.

Result


<table>
  <thead>
   <td><strong>names</strong>
   </td>
  <thead>
  <tr>
   <td>"D"
   </td>
  </tr>
  <tr>
   <td>"E"
   </td>
  </tr>
  <tr>
   <td>2 rows
   </td>
  </tr>
</table>

## Return middle two rows

To return a subset of the result, starting from somewhere in the middle, use this syntax:

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
	MATCH (n)
	RETURN n.name
	ORDER BY n.name
	SKIP 1
	LIMIT 2
$$) as (names agtype);
```

Two vertices from the middle are returned.

Result


<table>
  <thead>
   <td><strong>names</strong>
   </td>
  <thead>
  <tr>
   <td>"B"
   </td>
  </tr>
  <tr>
   <td>"C"
   </td>
  </tr>
  <tr>
   <td>2 rows
   </td>
  </tr>
</table>

## Using an expression with SKIP to return a subset of the rows

Using an expression with `SKIP` to return a subset of the rows

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
	MATCH (n)
	RETURN n.name
	ORDER BY n.name
	SKIP (3 * rand())+ 1
$$) as (a agtype);
```

The first two vertices are skipped, and only the last three are returned in the result.

Result


<table>
  <thead>
   <td><strong>names</strong>
   </td>
  <thead>
  <tr>
   <td>"C"
   </td>
  </tr>
  <tr>
   <td>"D"
   </td>
  </tr>
  <tr>
   <td>"E"
   </td>
  </tr>
  <tr>
   <td>3 rows
   </td>
  </tr>
</table>


# File: age-website-master/docs/clauses/with.md
# WITH

## Introduction

Using `WITH`, you can manipulate the output before it is passed on to the following query parts. The manipulations can be of the shape and/or number of entries in the result set.

`WITH` can also, like `RETURN`, alias expressions that are introduced into the results using the aliases as the binding name.

`WITH` is also used to separate the reading of the graph from updating of the graph. Every part of a query must be either read-only or write-only. When going from a writing clause to a reading clause, an optional `WITH` can be used to do so.


## Filter on aggregate function results

Aggregated results have to pass through a `WITH` clause to be able to filter on.

Query
```postgresql
SELECT *
FROM cypher('graph_name', $$
	MATCH (david {name: 'David'})-[]-(otherPerson)-[]->()
	WITH otherPerson, count(*) AS foaf
	WHERE foaf > 1
	RETURN otherPerson.name
$$) as (name agtype);
```


The name of the person connected to 'David' with the at least more than one outgoing relationship will be returned by the query.

Result
<table>
  <thead>
  <tr>
   <td>name
   </td>
  </tr>
  </thead>
  <tbody>
  <tr>
   <td>"Anders"
   </td>
  </tr>
  </tbody>
  <tr>
   <td>1 row
   </td>
  </tr>
</table>



## Sort results before using collect on them

You can sort results before passing them to collect, thus sorting the resulting list.

Query
```postgresql
SELECT *
FROM cypher('graph_name', $$
	MATCH (n)WITH n
	ORDER BY n.name DESC LIMIT 3
	RETURN collect(n.name)
$$) as (names agtype);
```


A list of the names of people in reverse order, limited to 3, is returned in a list.

Result
<table>
  <thead>
  <tr>
   <td>names
   </td>
  </tr>
  </thead>
  <tbody>
  <tr>
   <td>["Emil","David","Ceasar"]
   </td>
  </tr>
  </tbody>
  <tr>
   <td>1 row
   </td>
  </tr>
</table>

## Limit branching of a path search

You can match paths, limit to a certain number, and then match again using those paths as a base, as well as any number of similar limited searches.

Query

```postgresql
SELECT *
FROM cypher('graph_name', $$
	MATCH (n {name: 'Anders'})-[]-(m)WITH m
	ORDER BY m.name DESC LIMIT 1
	MATCH (m)-[]-(o)
	RETURN o.name
$$) as (name agtype);
```


Starting at 'Anders', find all matching nodes, order by name descending and get the top result, then find all the nodes connected to that top result, and return their names.

Result
<table>
  <thead>
  <tr>
   <td>name
   </td>
  </tr>
  </thead>
  <tbody>
  <tr>
   <td>"Anders"
   </td>
  </tr>
  <tr>
   <td>"Bossman"
   </td>
  </tr>
  </tbody>
  <tr>
   <td>2 rows
   </td>
  </tr>
</table>










# File: age-website-master/docs/functions/aggregate_functions.md
# Aggregation Functions

Functions that activate [auto aggregation](../intro/aggregation.md).

## Data Setup
```postgresql
LOAD 'age';
SET search_path TO ag_catalog;

SELECT create_graph('graph_name');

SELECT * FROM cypher('graph_name', $$
	CREATE (a:Person {name: 'A', age: 13}),
	(b:Person {name: 'B', age: 33, eyes: "blue"}),
	(c:Person {name: 'C', age: 44, eyes: "blue"}),
	(d1:Person {name: 'D', eyes: "brown"}),
	(d2:Person {name: 'D'}),
	(a)-[:KNOWS]->(b),
	(a)-[:KNOWS]->(c),
	(a)-[:KNOWS]->(d1),
	(b)-[:KNOWS]->(d2),
	(c)-[:KNOWS]->(d2)
$$) as (a agtype);
```

## min

`min()` returns the minimum value in a set of values.


Syntax: `min(expression)`

Returns:


```
A property type, or a list, depending on the values returned by expression.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An expression returning a set containing any combination of property types and lists thereof.
   </td>
  </tr>
</table>


Considerations:



* Any null values are excluded from the calculation.
* In a mixed set, any string value is always considered to be lower than any numeric value, and any list is always considered to be lower than any string.
* Lists are compared in dictionary order, i.e. list elements are compared pairwise in ascending order from the start of the list to the end.
* `min(null)` returns null.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (v:Person)
    RETURN min(v.age)
$$) as (min_age agtype);
```
The lowest of all the values in the property age is returned.

Result:


<table>
  <tr>
   <td>min_age
   </td>
  </tr>
  <tr>
   <td>13
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



### Using `min()` with Lists

Data Setup:

To clarify the following example, assume the next three commands are run first:


```postgresql
SELECT * FROM cypher('graph_name', $$ 
    CREATE (:min_test {val:'d'})
$$) as (result agtype);

SELECT * FROM cypher('graph_name', $$
    CREATE (:min_test {val:['a', 'b', 23]})
$$) as (result agtype);

SELECT * FROM cypher('graph_name', $$ 
    CREATE (:min_test {val:[1, 'b', 23]})
$$) as (result agtype);
```


Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (v:min_test)
    RETURN min(v.val)
$$) as (min_val agtype);
```


The lowest of all the values in the set—in this case, the list ['a', 'b', 23]—is returned, as (i) the two lists are considered to be lower values than the string "d", and (ii) the string "a" is considered to be a lower value than the numerical value 1.

Result:


<table>
  <tr>
   <td>min_age
   </td>
  </tr>
  <tr>
   <td>["a", "b", 23]
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## max

`max()` returns the maximum value in a set of values.

Syntax: `max(expression)`

Returns:


```
A property type, or a list, depending on the values returned by expression.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An expression returning a set containing any combination of property types and lists thereof.
   </td>
  </tr>
</table>


Considerations:



* Any null values are excluded from the calculation.
* In a mixed set, any numeric value is always considered to be higher than any string value, and any string value is always considered to be higher than any list.
* Lists are compared in dictionary order, i.e. list elements are compared pairwise in ascending order from the start of the list to the end.
* `max(null)` returns null.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (n:Person)
    RETURN max(n.age)
$$) as (max_age agtype);
```


The highest of all the values in the property age is returned.

Result:


<table>
  <tr>
   <td>min_age
   </td>
  </tr>
  <tr>
   <td>44
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## stDev

`stDev()` returns the standard deviation for the given value over a group. It uses a standard two-pass method, with N - 1 as the denominator, and should be used when taking a sample of the population for an unbiased estimate. When the standard deviation of the entire population is being calculated, `stDevP` should be used.

Syntax: `stDev(expression)`

Returns: 


```
An agtype float.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype number expression
   </td>
  </tr>
</table>


Considerations:



* Any null values are excluded from the calculation.
* `stDev(null)` returns 0.0 (zero).

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
   MATCH (n:Person)
   RETURN stDev(n.age)
$$) as (stdev_age agtype);
```


The standard deviation of the values in the property age is returned.

Result:


<table>
  <tr>
   <td>stdev_age
   </td>
  </tr>
  <tr>
   <td>15.716233645501712
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## stDevP

`stDevP()` returns the standard deviation for the given value over a group. It uses a standard two-pass method, with N as the denominator, and should be used when calculating the standard deviation for an entire population. When the standard deviation of only a sample of the population is being calculated, `stDev` should be used.

Syntax: `stDevP(expression)`

Returns:


```
An agtype float.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype number expression
   </td>
  </tr>
</table>


Considerations:



* Any null values are excluded from the calculation.
* `stDevP(null)` returns 0.0 (zero).

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (n:Person)
    RETURN stDevP(n.age)
$$) as (stdevp_age agtype);
```


The population standard deviation of the values in the property age is returned. 

Result:


<table>
  <tr>
   <td>stdevp_age
   </td>
  </tr>
  <tr>
   <td>12.832251036613439
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## percentileCont

`percentileCont()` returns the percentile of the given value over a group, with a percentile from 0.0 to 1.0. It uses a linear interpolation method, calculating a weighted average between two values if the desired percentile lies between them. For nearest values using a rounding method, see `percentileDisc`.

Syntax: `percentileCont(expression, percentile)`

Returns:


```
An agtype float.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype number expression
   </td>
  </tr>
  <tr>
   <td>percentile
   </td>
   <td>An agtype number value between 0.0 and 1.0
   </td>
  </tr>
</table>


Considerations:



* Any null values are excluded from the calculation.
* `percentileCont(null, percentile)` returns null.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (n:Person)
    RETURN percentileCont(n.age, 0.4)
$$) as (percentile_cont_age agtype);
```


The 40th percentile of the values in the property age is returned, calculated with a weighted average. In this case, 0.4 is the median, or 40th percentile.

Result:


<table>
  <tr>
   <td>percentile_cont_age
   </td>
  </tr>
  <tr>
   <td>29.0
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## percentileDisc

`percentileDisc()` returns the percentile of the given value over a group, with a percentile from 0.0 to 1.0. It uses a rounding method and calculates the nearest value to the percentile. For interpolated values, see `percentileCont`.

Syntax: `percentileDisc(expression, percentile)`

Returns:


```
An agtype float.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype number expression
   </td>
  </tr>
  <tr>
   <td>percentile
   </td>
   <td>An agtype number value between 0.0 and 1.0
   </td>
  </tr>
</table>


Considerations:



* Any null values are excluded from the calculation.
* `percentileDisc(null, percentile)` returns null.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (n:Person)
    RETURN percentileDisc(n.age, 0.5)
$$) as (percentile_disc_age agtype);
```


The 50th percentile of the values in the property age is returned. 

Result:


<table>
  <tr>
   <td>percentile_cont_age
   </td>
  </tr>
  <tr>
   <td>33.0
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## count

`count()` returns the number of values or records, and appears in two variants:



* `count(*)` returns the number of matching records
* `count(expr)` returns the number of non-null values returned by an expression.

Syntax: `count(expression)`

Returns:


```
An agtype integer.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An expression
   </td>
  </tr>
</table>


Considerations:
* `count(*)` includes records returning null.
* `count(expr)` ignores null values.
* `count(null)` returns 0 (zero).
* `count(*)` can be used to return the number of nodes; for example, the number of nodes connected to some node n.


Query
```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (n {name: 'A'})-[]->(x)
    RETURN n.age, count(*)
$$) as (age agtype, number_of_people agtype);
```

The age property of the start node n (with a name value of 'A') and the number of nodes related to n are returned.

Result:
<table>
  <tr>
   <td>age
   </td>
   <td>number_of_people
   </td>
  </tr>
  <tr>
   <td>13
   </td>
   <td>3
   </td>
  </tr>
  <tr>
   <td colspan="2" >1 row(s) returned
   </td>
  </tr>
</table>


Using `count(*)` can be used to group and count relationship types, returning the number of relationships of each type.

Query
```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (n {name: 'A'})-[r]->()
    RETURN type(r), count(*)
$$) as (label agtype, count agtype);
```


The relationship type and the number of relationships with that type are returned.

Result:


<table>
  <tr>
   <td>label
   </td>
   <td>count
   </td>
  </tr>
  <tr>
   <td>“KNOWS”
   </td>
   <td>3
   </td>
  </tr>
  <tr>
   <td colspan="2" >1 row(s) returned
   </td>
  </tr>
</table>



### Using `count(expression)` to return the number of values

Instead of simply returning the number of records with `count(*)`, it may be more useful to return the actual number of values returned by an expression.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (n {name: 'A'})-[]->(x)
    RETURN count(x)
$$) as (count agtype);
```


The number of nodes connected to the start node n is returned.

Result:


<table>
  <tr>
   <td>count
   </td>
  </tr>
  <tr>
   <td>3
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



### Counting non-null values

`count(expression)` can be used to return the number of non-null values returned by the expression.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (n:Person)
    RETURN count(n.age)
$$) as (count agtype);
```


The number of nodes with the label `Person` that have a non-null value for the age property is returned.

Result:


<table>
  <tr>
   <td>count
   </td>
  </tr>
  <tr>
   <td>3
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>


### Counting with and without duplicates

In this example we are trying to find all our friends of friends, and count them:
* The first aggregate function, `count(DISTINCT friend_of_friend)`, will only count a `friend_of_friend` once, as `DISTINCT` removes the duplicates.
* The second aggregate function, `count(friend_of_friend)`, will consider the same `friend_of_friend` multiple times.

Query
```postgresql
SELECT *
FROM cypher('graph_name', $$
	MATCH (me:Person)-[]->(friend:Person)-[]->(friend_of_friend:Person)
	WHERE me.name = 'A'
	RETURN count(DISTINCT friend_of_friend), count(friend_of_friend)
$$) as (friend_of_friends_distinct agtype, friend_of_friends agtype);
```

Both B and C know D and thus D will get counted twice when not using `DISTINCT`.

Result:
<table>
  <tr>
   <td>friend_of_friends_distinct
   </td>
   <td>friend_of_friends
   </td>
  </tr>
  <tr>
   <td>1
   </td>
   <td>2
   </td>
  </tr>
  <tr>
   <td>1 row
   </td>
  </tr>
</table>


## avg

`avg()` returns the average of a set of numeric values.

Syntax: `avg(expression)`

Returns:


```
An agtype integer
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An expression returning a set of numeric values.
   </td>
  </tr>
</table>


Considerations:



* Any null values are excluded from the calculation.
* `avg(null)` returns null.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
MATCH (n:Person)
RETURN avg(n.age)
$$) as (avg_age agtype);
```


The average of all the values in the property age is returned. 

Result:


<table>
  <tr>
   <td>avg_age
   </td>
  </tr>
  <tr>
   <td>30.0
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## sum 

`sum()` returns the sum of a set of numeric values.

Syntax: `sum(expression)`

Returns:


```
An agtype float
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An expression returning a set of numeric values.
   </td>
  </tr>
</table>


Considerations:



* Any null values are excluded from the calculation.
* `sum(null)` returns null.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
MATCH (n:Person)
RETURN sum(n.age)
$$) as (total_age agtype);
```


The sum of all the values in the property age is returned.

Result:


<table>
  <tr>
   <td>total_age
   </td>
  </tr>
  <tr>
   <td>90
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>


# File: age-website-master/docs/functions/list_functions.md
# List Functions 

## Data Setup

```postgresql
SELECT * from cypher('graph_name', $$
CREATE (A:Person {name: 'Alice', age: 38, eyes: 'brown'}),
	(B:Person {name: 'Bob', age: 25, eyes: 'blue'}),
	(C:Person {name: 'Charlie', age: 53, eyes: 'green'}),
	(D:Person {name: 'Daniel', age: 54, eyes: 'brown'}),
	(E:Person {name: 'Eskil', age: 41, eyes: 'blue', array: ['one', 'two', 'three']}),
	(A)-[:KNOWS]->(B),
	(A)-[:KNOWS]->(C),
	(B)-[:KNOWS]->(D),
	(C)-[:KNOWS]->(D),
	(B)-[:KNOWS]->(E)
$$) as (result agtype);
```

## keys

`keys()` returns a list containing the string representations for all the property names of a vertex, edge, or map.

Syntax: `keys(expression)`

Returns:
```
An agtype list containing string agtype elements
```

Arguments:
<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>path
   </td>
   <td>An expression that returns a vertex, edge, or map.
   </td>
  </tr>
</table>

Considerations:
* `keys(null)` returns null.

Query:
```postgresql
SELECT * from cypher('graph_name', $$
	MATCH (a)
	WHERE a.name = 'Alice'
	RETURN keys(a)
$$) as (result agtype);
```

A list containing the names of all the properties on the vertex bound to `a` is returned.

Result:


<table>
  <tr>
   <td>keys
   </td>
  </tr>
  <tr>
   <td>["age", "eyes", "name"]</td>
  </tr>
  <tr>
   <td colspan="1" >1 rows
   </td>
  </tr>
</table>

## range

`range()` returns a list comprising all integer values within a range bounded by a start value **start** and end value **end**, where the difference **step** between any two consecutive values is constant; i.e. an arithmetic progression. The range is  inclusive, and the arithmetic progression will therefore always contain **start** and—depending on the values of **start**, **step** and **end**—**end**.

Syntax: `range(start, end [, step])`

Returns:
```
An agtype list containing integer elements
```

Arguments:
<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>start
   </td>
   <td>An expression that returns an integer value.
   </td>
  </tr>
  <tr>
   <td>end
   </td>
   <td>An expression that returns an integer value.
   </td>
  </tr>
  <tr>
   <td>step
   </td>
   <td>A numeric expression defining the difference between any two consecutive values, with a default of 1.
   </td>
  </tr>
</table>

Query:
```postgresql
SELECT *
FROM cypher('graph_name', $$
	RETURN range(0, 10), range(2, 18, 3)
$$) as (no_step agtype, step agtype);
```

Two lists of numbers in the given ranges are returned.

Result:
<table>
  <tr>
   <td>no_step
   </td>
   <td>step
   </td>
  </tr>
  <tr>
   <td>[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]</td>
   <td>[2, 5, 8, 11, 14, 17]</td>
  </tr>
  <tr>
   <td colspan="1" >1 row
   </td>
  </tr>
</table>

## labels

`labels` returns a list containing the string representations for all the labels of a node.

Syntax: `labels(vertex)`

Returns:
```
An agtype list containing string elements
```

Arguments:
<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>vertex
   </td>
   <td>An expression that returns a single vertex.
   </td>
  </tr>
</table>

Considerations:
* `labels(null)` returns `null`.

Query:
```postgresql
SELECT *
FROM cypher('graph_name', $$
	MATCH (a)
	WHERE a.name = 'Alice'
	RETURN labels(a)
$$) as (edges agtype);
```

A list containing all the labels of the node bound to `a` is returned.

Result:
<table>
  <tr>
   <td>edges
   </td>
  </tr>
  <tr>
   <td>["Person"]
   </td>
  </tr>
  <tr>
   <td colspan="3" >1 row
   </td>
  </tr>
</table>

## nodes

`nodes` returns a list containing all the vertices in a path.

Syntax: `nodes(path)`

Returns:
```
An agtype list containing vertex entities
```

Arguments:
<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>path
   </td>
   <td>An expression that returns an agtype path.
   </td>
  </tr>
</table>

Considerations:
* `nodes(null)` returns `null`.

Query:
```postgresql
SELECT *
FROM cypher('graph_name', $$
	MATCH p = (a)-[]->(b)-[]->(c)
	WHERE a.name = 'Alice' AND c.name = 'Eskil'
	RETURN nodes(a)
$$) as (vertices agtype);
```

A list containing all the vertices in the path `p` is returned.

Result:
<table>
  <tr>
   <td>vertices
   </td>
  </tr>
  <tr>
   <td> [{"id": 844424930131969, "label": "Person", "properties": {"age": 38, "eyes": "brown", "name": "Alice"}}::vertex, {"id": 844424930131970, "label": "Person", "properties": {"age": 25, "eyes": "blue", "name": "Bob"}}::vertex, {"id": 844424930131973, "label": "Person", "properties": {"age": 41, "eyes": "blue", "name": "Eskil", "array": ["one", "two", "three"]}}::vertex]
   </td>
  </tr>
  <tr>
   <td colspan="3" >1 row
   </td>
  </tr>
</table>

## relationships

`relationships()` returns a list containing all the relationships in a path.

Syntax: `relationships(path)`

Returns:
```
An agtype list containing edge entities
```

Arguments:
<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>path
   </td>
   <td>An expression that returns an agtype path.
   </td>
  </tr>
</table>

Considerations:
* `relationships(null)` returns `null`.

Query:
```postgresql
SELECT *
FROM cypher('graph_name', $$
	MATCH p = (a)-[]->(b)-[]->(c)
	WHERE a.name = 'Alice' AND c.name = 'Eskil'
	RETURN relationships(p)
$$) as (edges agtype);
```

A list containing all the edges in the path `p` is returned.

Result:
<table>
  <tr>
   <td>edges
   </td>
  </tr>
  <tr>
   <td>[{"id": 1125899906842640, "label": "KNOWS", "end_id": 844424930131989, "start_id": 844424930131988, "properties": {}}::edge, {"id": 1125899906842644, "label": "KNOWS", "end_id": 844424930131992, "start_id": 844424930131989, "properties": {}}::edge]
   </td>
  </tr>
  <tr>
   <td colspan="3" >1 row
   </td>
  </tr>
</table>

## toBooleanList
`toBooleanList()` converts a list of values and returns a list of boolean values. If any values are not convertible to boolean they will be null in the list returned.

Syntax: `toBooleanList(list)`

Returns:
```
An agtype list containing the converted elements; depending on the input value a converted value is either a boolean value or null.
```

Considerations:
* Any null element in list is preserved.
* Any boolean value in list is preserved.
* If the list is null, null will be returned.
* If the list is not a list, an error will be returned.

Query:
```postgresql
SELECT * FROM cypher('expr', $$
    RETURN toBooleanList(["true", "false", "true"])
$$) AS (toBooleanList agtype);
```

Result:
<table>
  <tr>
   <td>tobooleanlist
   </td>
  </tr>
  <tr>
   <td> [true, false, true]
   </td>
  </tr>
  <tr>
   <td colspan="3" >1 row
   </td>
  </tr>
</table>


# File: age-website-master/docs/functions/logarithmic_functions.md
# Logarithmic Functions


## e

`e()` returns the base of the natural logarithm, e.

Syntax: `e()`

Returns:


```
An agtype float.
```


Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN e()
$$) as (e agtype);
```


Results


<table>
  <tr>
   <td>e
   </td>
  </tr>
  <tr>
   <td> 2.71828182845905
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## sqrt

`sqrt()` returns the square root of a number.

Syntax: `sqrt(expression)`

Returns:


```
An agtype float.
```


Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN sqrt(144)
$$) as (results agtype);
```


Results


<table>
  <tr>
   <td>results
   </td>
  </tr>
  <tr>
   <td>12
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## exp

`exp()` returns e^n, where e is the base of the natural logarithm, and n is the value of the argument expression.

Syntax: `exp(expression)`

Returns:


```
An agtype float.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype number expression
   </td>
  </tr>
</table>


Considerations:



* `exp(null)` returns `null`.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN exp(2)
$$) as (e agtype);
```


e to the power of 2 is returned.

Result:


<table>
  <tr>
   <td>e
   </td>
  </tr>
  <tr>
   <td>7.38905609893065
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## log

`log()` returns the natural logarithm of a number.

Syntax: `log(expression)`

Returns:


```
An agtype float.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype number expression
   </td>
  </tr>
</table>


Considerations:



* `log(null)` returns `null`.
* `log(0)` returns `null`.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN log(27)
$$) as (natural_logarithm agtype);
```


The natural logarithm of 27 is returned.

Result:


<table>
  <tr>
   <td>natural_logarithm
   </td>
  </tr>
  <tr>
   <td>3.295836866004329
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## log10

`log10()` returns the common logarithm (base 10) of a number.

Syntax: `log10(expression)`

Returns:


```
An agtype float.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype number expression
   </td>
  </tr>
</table>


Considerations:



* `log10(null)` returns `null`.
* `log10(0)` returns `null`.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN log10(27)
$$) as (common_logarithm agtype);
```


The common logarithm of 27 is returned.

Result:


<table>
  <tr>
   <td>common_logarithm
   </td>
  </tr>
  <tr>
   <td>1.4313637641589874
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



# File: age-website-master/docs/functions/map_functions.md
# Map Functions

In AGE, a map is a data structure that allows you to store a collection of key-value pairs. Each key within a map is unique, and is associated with a corresponding value.
This data structure is similar to dictionaries in Python or objects in JavaScript, providing an efficient way to organize and retrieve data based on keys.
This section focuses on explaining various functions that allow you to generate and manipulate maps effectively.

## vertex_stats()
The `vertex_stats()` function can extract information from a vertex. Upon passing a vertex as an argument to the vertex_stats function, 
you are presented with a structured map, which includes the following key-value pairs:

* id: A unique identifier assigned to the vertex;
* label: The label or type that categorizes the vertex;
* in_degree: The count of incoming edges directed towards the vertex;
* out_degree: The count of outgoing edges originating from the vertex;
* self_loops: The count of self-loop edges associated with the vertex.

Syntax: `vertex_stats(vertex)`

### Setup

```sql
-- Creating the graph.
SELECT create_graph('vertex_stats_graph');

-- Creating vertices and edges.
SELECT * FROM cypher('vertex_stats_graph', $$
CREATE (:Person {name: 'John Donne'})-[:WROTE]->(:Poem {title: 'Holy Sonnet XIV'})
$$) AS (a agtype);
```

### Query

```sql
SELECT * FROM cypher('vertex_stats_graph', $$
MATCH (v:Poem {title: 'Holy Sonnet XIV'})
RETURN vertex_stats(v)
$$) AS (vertex_stats agtype);
```

### Result

<table>
  <tr>
   <td>vertex_stats
   </td>
  </tr>
  <tr>
   <td>{"id": 1407374883553281, "label": "Poem", "in_degree": 1, "out_degree": 0, "self_loops": 0}
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>

### Retrieving Values

It is also possible to retrieve specific values from the generated map using the following syntax: 

`vertex_stats(vertex)["key"]`


# File: age-website-master/docs/functions/numeric_functions.md
# Numeric Functions


## rand

`rand()` returns a random floating point number in the range from 0 (inclusive) to 1 (exclusive); i.e.[0,1). The numbers returned follow an approximate uniform distribution.

Syntax: `rand()`

Returns:


```
A float.
```


Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN rand()
$$) as (random_number agtype);
```


A random number is returned.

Result:


<table>
  <tr>
   <td>random_number
   </td>
  </tr>
  <tr>
   <td>0.3586784748902053
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## abs

`abs()` returns the absolute value of the given number.

Syntax: `abs(expression)`

Returns:


```
The type of the value returned will be that of expression.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype number expression
   </td>
  </tr>
</table>


Considerations:



* `abs(null)` returns null.
* If expression is negative, -(expression) (i.e. the negation of expression) is returned.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (a), (e) WHERE a.name = 'Alice' AND e.name = 'Eskil'
    RETURN a.age, e.age, abs(a.age - e.age)
$$) as (alice_age agtype, eskil_age agtype, difference agtype);
```


The absolute value of the age difference is returned.

Result:


<table>
  <tr>
   <td>alice_age
   </td>
   <td>eskil_age
   </td>
   <td>difference
   </td>
  </tr>
  <tr>
   <td>38
   </td>
   <td>41
   </td>
   <td>3
   </td>
  </tr>
  <tr>
   <td colspan="3" >1 row(s) returned
   </td>
  </tr>
</table>



## ceil

`ceil()` returns the smallest floating point number that is greater than or equal to the given number and equal to a mathematical integer.

Syntax: `ceil(expression)`

Returns:


```
A float.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype number expression
   </td>
  </tr>
</table>


Considerations:



* `ceil(null)` returns `null`.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN ceil(0.1)
$$) as (ceil_value agtype);
```


The ceiling of 0.1 is returned.

Result:


<table>
  <tr>
   <td> ceil_value
   </td>
  </tr>
  <tr>
   <td>1
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## floor

`floor()` returns the greatest floating point number that is less than or equal to the given number and equal to a mathematical integer.

Syntax: `floor(expression)`

Returns:


```
A float.
```


 

Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype number expression
   </td>
  </tr>
</table>


Considerations:



* `floor(null)` returns null.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN floor(0.1)
$$) as (flr agtype);
```


The floor of 0.1 is returned.

Result:


<table>
  <tr>
   <td>flr
   </td>
  </tr>
  <tr>
   <td>0
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## round

`round()` returns the value of the given number rounded to the nearest integer.

Syntax: `round(expression)`

Returns:


```
A float.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype number expression
   </td>
  </tr>
</table>

Considerations:



* `round(null)` returns `null`.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN round(3.141592)
$$) as (rounded_value agtype);
```


3.0 is returned.

Result:


<table>
  <tr>
   <td>rounded_value
   </td>
  </tr>
  <tr>
   <td>3.0
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## sign

`sign()` returns the signum of the given number: 0 if the number is 0, -1 for any negative number, and 1 for any positive number

Syntax: `sign(expression)`

Returns:

```
An integer.
```



Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype number expression
   </td>
  </tr>
</table>


Considerations:



* `sign(null)` returns `null`.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN sign(-17), sign(0.1), sign(0)
$$) as (negative_sign agtype, positive_sign agtype, zero_sign agtype);
```


The signs of -17 and 0.1 are returned.

Result:


<table>
  <tr>
   <td>negative_sign
   </td>
   <td>positive_sign
   </td>
   <td>zero_sign
   </td>
  </tr>
  <tr>
   <td>-1
   </td>
   <td>1
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td colspan="3" >1 row(s) returned
   </td>
  </tr>
</table>






# File: age-website-master/docs/functions/predicate_functions.md
# Predicate Functions

Predicates are boolean functions that return true or false for a given set of input. They are most commonly used to filter out subgraphs in the WHERE part of a query.


## Exists(Property)

`exists()` returns `true` if the specified property exists in the node, relationship or map. This is different from the `EXISTS` clause.

Syntax: `exists(property)`

Returns:

An agtype boolean

Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>property
   </td>
   <td>A property from a vertex or edge
   </td>
  </tr>
</table>


Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
     MATCH (n)
     WHERE exists(n.surname)
     RETURN n.first_name, n.last_name
$$) as (first_name agtype, last_name agtype);
```


Results:


<table>
  <tr>
   <td>first_name
   </td>
   <td>last_name
   </td>
  </tr>
  <tr>
   <td>‘John
   </td>
   <td>‘Smith’
   </td>
  </tr>
  <tr>
   <td>‘Patty’
   </td>
   <td>‘Patterson’
   </td>
  </tr>
  <tr>
   <td colspan="2" >2 row(s) returned
   </td>
  </tr>
</table>


## Exists(Path)

`EXISTS(path)` returns `true` if for the given path, there already exists the given path.

```postgresql
SELECT *
FROM cypher('graph_name', $$
     MATCH (n)
     WHERE exists((n)-[]-({name: 'Willem Defoe'}))
     RETURN n.full_name
$$) as (full_name agtype);
```

Results:
<table>
  <tr>
   <td>full_name
   </td>
  </tr>
  <tr>
   <td>‘Toby Maguire'
   </td>
  </tr>
  <tr>
   <td>‘Tom Holland’
   </td>
  </tr>
  <tr>
   <td colspan="2" >2 row(s) returned
   </td>
  </tr>
</table>



# File: age-website-master/docs/functions/scalar_functions.md
# Scalar Functions 


## id

`id()` returns the id of a vertex or edge.

Syntax: `id(expression)`

Returns:


```
An agtype integer
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An expression that returns a vertex or edge.
   </td>
  </tr>
</table>


Considerations:

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (a)
    RETURN id(a)
$$) as (id agtype);
```


Results


<table>
  <tr>
   <td>id
   </td>
  </tr>
  <tr>
   <td>0
   </td>
  </tr>
  <tr>
   <td>1
   </td>
  </tr>
  <tr>
   <td>2
   </td>
  </tr>
  <tr>
   <td>3
   </td>
  </tr>
  <tr>
   <td>4 row(s) returned
   </td>
  </tr>
</table>



## start_id

`start_id()` returns the id of the vertex that is the starting vertex for the edge.

Syntax: `start_id(expression)`

Returns:


```
An agtype integer
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An expression that evaluates to an edge.
   </td>
  </tr>
</table>


Considerations:

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH ()-[e]->()
    RETURN start_id(e)
$$) as (start_id agtype);
```


Results


<table>
  <tr>
   <td>start_id
   </td>
  </tr>
  <tr>
   <td>0
   </td>
  </tr>
  <tr>
   <td>1
   </td>
  </tr>
  <tr>
   <td>2
   </td>
  </tr>
  <tr>
   <td>3
   </td>
  </tr>
  <tr>
   <td>4 row(s) returned
   </td>
  </tr>
</table>



## end_id

`end_id()` returns the id of the vertex that is the ending vertex for the edge.

Syntax: `end_id(expression)`

Returns:


```
An agtype integer
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An expression that evaluates to an edge.
   </td>
  </tr>
</table>


Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH ()-[e]->()
    RETURN end_id(e)
$$) as (end_id agtype);
```


Results


<table>
  <tr>
   <td>end_id
   </td>
  </tr>
  <tr>
   <td>4
   </td>
  </tr>
  <tr>
   <td>5
   </td>
  </tr>
  <tr>
   <td>6
   </td>
  </tr>
  <tr>
   <td>7
   </td>
  </tr>
  <tr>
   <td>4 row(s) returned
   </td>
  </tr>
</table>



## type

`type()` returns the string representation of the edge type.

Syntax: `type(edge)`

Returns:


```
An agtype string
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>edge
   </td>
   <td>An expression that evaluates to an edge.
   </td>
  </tr>
</table>


Considerations:

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH ()-[e]->()
    RETURN type(e)
$$) as (type agtype);
```


Results


<table>
  <tr>
   <td>type
   </td>
  </tr>
  <tr>
   <td>“KNOWS”
   </td>
  </tr>
  <tr>
   <td>“KNOWS”
   </td>
  </tr>
  <tr>
   <td>2 row(s) returned
   </td>
  </tr>
</table>



## properties

Returns an agtype map containing all the properties of a vertex or edge. If the argument is already a map, it is returned unchanged.

Syntax: `properties(expression)`

Returns:


```
An agtype map.
```


Arguments:

<table>
   <tr>
      <td>Name
      </td>
      <td>Description
      </td>
   </tr>
   <tr>
      <td>Expression
      </td>
      <td>An expression that returns a vertex, an edge, or an agtype map.
      </td>
   </tr>
</table>

Considerations: 

* `properties(null)` returns `null`.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    CREATE (p:Person {name: 'Stefan', city: 'Berlin'})
    RETURN properties(p)
$$) as (type agtype);
```


Results:


<table>
  <tr>
   <td><strong>properties</strong>
   </td>
  </tr>
  <tr>
   <td>{name: "Stefan"; city: "Berlin"}
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## head

returns the first element in an agtype list.

Syntax: `head(list)`

Returns:


```
The type of the value returned will be that of the first element of the list.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>List
   </td>
   <td>An expression that returns a list
   </td>
  </tr>
</table>


Considerations:



* `head(null)` returns `null`.
* If the first element in the list is `null`, `head(list)` will return `null`.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
   MATCH (a)
   WHERE a.name = 'Eskil'
   RETURN a.array, head(a.array)
$$) as (lst agtype, lst_head agtype);
```


The first element in the list is returned.

Result:


<table>
  <tr>
   <td>lst
   </td>
   <td>lst_head
   </td>
  </tr>
  <tr>
   <td>["one","two","three"]
   </td>
   <td>"one"
   </td>
  </tr>
  <tr>
   <td colspan="2" >1 row(s) returned
   </td>
  </tr>
</table>



## last

returns the last element in an agtype list.

Syntax: `last(list)`

Returns:


```
The type of the value returned will be that of the last element of the list.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>List
   </td>
   <td>An expression that returns a list
   </td>
  </tr>
</table>


Considerations:



* `tail(null)` returns `null`.
* If the last element in the list is `null`, `last(list)` will return `null`.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
MATCH (a)
WHERE a.name = 'Eskil'
RETURN a.array, last(a.array)
$$) as (lst agtype, lst_tail agtype);
```


The first element in the list is returned.

Result:


<table>
  <tr>
   <td>lst
   </td>
   <td>lst_tail
   </td>
  </tr>
  <tr>
   <td>["one","two","three"]
   </td>
   <td>"three"
   </td>
  </tr>
  <tr>
   <td colspan="2" >1 row(s) returned
   </td>
  </tr>
</table>



## length

`length()` returns the length of a path.

Syntax: `length(path)`

Returns:


```
An agtype integer.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>path
   </td>
   <td>An expression that returns a path.
   </td>
  </tr>
</table>


Considerations: `length(null)` returns `null`.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
   MATCH p = (a)-[]->(b)-[]->(c)
   WHERE a.name = 'Alice'
   RETURN length(p)
$$) as (length_of_path agtype);
```


The length of the path `p` is returned.

Results:


<table>
  <tr>
   <td>length_of_path
   </td>
  </tr>
  <tr>
   <td>2
   </td>
  </tr>
  <tr>
   <td>2
   </td>
  </tr>
  <tr>
   <td>2
   </td>
  </tr>
  <tr>
   <td>3 row(s) returned
   </td>
  </tr>
</table>



## size

`size()` returns the length of a list.

Syntax: `size(list)`

Returns:


```
An agtype integer.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>list
   </td>
   <td>An expression that returns a list.
   </td>
  </tr>
</table>


Considerations:



* `size(null)` returns `null`.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN size(['Alice', 'Bob'])
$$) as (size_of_list agtype);
```


The length of the path `p` is returned.

Results:


<table>
  <tr>
   <td>size_of_list
   </td>
  </tr>
  <tr>
   <td>2
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## startNode

`startNode()` returns the start node of an edge.

Syntax: `startNode(edge)`

Returns:


```
A vertex.
```


Arguments:


<table>
  <tr>
   <td>Name 
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>edge
   </td>
   <td>An expression that returns an edge.
   </td>
  </tr>
</table>


Considerations:



* `startNode(null)` returns `null`.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (x:Developer)-[r]-()
    RETURN startNode(r)
$$) as (v agtype);
```


Result


<table>
  <tr>
   <td>v
   </td>
  </tr>
  <tr>
   <td>Node[0]{name:"Alice",age:38,eyes:"brown"}
   </td>
  </tr>
  <tr>
   <td>Node[0]{name:"Alice",age:38,eyes:"brown"}
   </td>
  </tr>
  <tr>
   <td>2 row(s) returned
   </td>
  </tr>
</table>



## endNode

`endNode()` returns the start node of an edge.

Syntax: `endNode(edge)`

Returns:


```
A vertex.
```


Arguments:


<table>
  <tr>
   <td>Name 
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>edge
   </td>
   <td>An expression that returns an edge.
   </td>
  </tr>
</table>


Considerations:



* `endNode(null)` returns `null`.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    MATCH (x:Developer)-[r]-()
    RETURN endNode(r)
$$) as (v agtype);
```


Result


<table>
  <tr>
   <td>v
   </td>
  </tr>
  <tr>
   <td>Node[2]{name:"Charlie",age:53,eyes:"green"}
   </td>
  </tr>
  <tr>
   <td>Node[1]{name:"Bob",age:25,eyes:"blue"}
   </td>
  </tr>
  <tr>
   <td>2 row(s) returned
   </td>
  </tr>
</table>



## timestamp

`timestamp()` returns the difference, measured in milliseconds, between the current time and midnight, January 1, 1970 UTC.

Syntax: `timestamp()`

Returns:


```
An agtype integer.
```


Considerations:



* `timestamp()` will return the same value during one entire query, even for long-running queries.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN timestamp()
$$) as (t agtype);
```


The time in milliseconds is returned.

Results:


<table>
  <tr>
   <td><strong>t</strong>
   </td>
  </tr>
  <tr>
   <td>1613496720760
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## toBoolean

`toBoolean()` converts a string value to a boolean value.

Syntax: `toBoolean(expression)`

Returns:


```
An agtype boolean.
```


Arguments:


<table>
  <tr>
   <td><strong>Name</strong>
   </td>
   <td><strong>Description</strong>
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An expression that returns a boolean or string value.
   </td>
  </tr>
</table>


Considerations:



* `toBoolean(null)` returns `null`.
* If expression is a boolean value, it will be returned unchanged.
* If the parsing fails, `null` will be returned.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN toBoolean('TRUE'), toBoolean('not a boolean')
$$) as (a_bool agtype, not_a_bool agtype);
```


Result:


<table>
  <tr>
   <td><strong>a_bool</strong>
   </td>
   <td><strong>not_a_bool</strong>
   </td>
  </tr>
  <tr>
   <td>true
   </td>
   <td>NULL
   </td>
  </tr>
  <tr>
   <td colspan="2" >1 row(s) returned
   </td>
  </tr>
</table>



## toFloat

`toFloat()` converts an integer or string value to a floating point number.

Syntax: `toFloat(expression)`

Returns:


```
A float.
```


<table>
  <tr>
   <td><strong>Name</strong>
   </td>
   <td><strong>Description</strong>
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An expression that returns an agtype number or agtype string value.
   </td>
  </tr>
</table>


Considerations:



* `toFloat(null)` returns `null`.
* If expression is a floating point number, it will be returned unchanged.
* If the parsing fails, `null` will be returned.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN toFloat('11.5'), toFloat('not a number')
$$) as (a_float agtype, not_a_float agtype);
```


Result:


<table>
  <tr>
   <td>a_float
   </td>
   <td>not_a_float
   </td>
  </tr>
  <tr>
   <td>11.5
   </td>
   <td>NULL
   </td>
  </tr>
  <tr>
   <td colspan="2" >1 row(s) returned
   </td>
  </tr>
</table>



## toInteger

`toInteger()` converts a floating point or string value to an integer value.

Syntax: `toInteger(expression)`

Returns:


```
An agtype integer.
```


Arguments


<table>
  <tr>
   <td><strong>Name</strong>
   </td>
   <td><strong>Description</strong>
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An expression that returns an agtype number or agtype string value.
   </td>
  </tr>
</table>


Considerations:



* `toInteger(null)` returns `null`.
* If expression is an integer value, it will be returned unchanged.
* If the parsing fails, `null` will be returned.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
     RETURN toInteger('42'), toInteger('not a number')
$$) as (an_integer agtype, not_an_integer agtype);
```


Result:


<table>
  <tr>
   <td>an_integer
   </td>
   <td>not_an_integer
   </td>
  </tr>
  <tr>
   <td>42
   </td>
   <td>NULL
   </td>
  </tr>
  <tr>
   <td colspan="2" >1 row(s) returned
   </td>
  </tr>
</table>



## coalesce

`coalesce()` returns the first non-null value in the given list of expressions.

Syntax:`coalesce(expression [, expression]*)`

Returns:


```
The type of the value returned will be that of the first non-null expression.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An expression which may return null.
   </td>
  </tr>
</table>


Considerations:



* `null` will be returned if all the arguments are null.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
MATCH (a)
WHERE a.name = 'Alice'
RETURN coalesce(a.hairColor, a.eyes), a.hair_color, a.eyes
$$) as (color agtype, hair_color agtype, eyes agtype);
```

Result


<table>
  <tr>
   <td>color
   </td>
   <td>hair_color
   </td>
   <td>eyes
   </td>
  </tr>
  <tr>
   <td>“brown”
   </td>
   <td>NULL
   </td>
   <td>“Brown”
   </td>
  </tr>
  <tr>
   <td colspan="3" >1 row(s) returned
   </td>
  </tr>
</table>




# File: age-website-master/docs/functions/string_functions.md
# String Functions


## replace

`replace()` returns a string in which all occurrences of a specified string in the original string have been replaced by another (specified) string.

Syntax: <code>replace(original, search, replace)</code></strong>

Returns:


```
An agtype string.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>original
   </td>
   <td>An expression that returns a string.
   </td>
  </tr>
  <tr>
   <td>search
   </td>
   <td>An expression that specifies the string to be replaced in original.
   </td>
  </tr>
  <tr>
   <td>replace
   </td>
   <td>An expression that specifies the replacementstring.
   </td>
  </tr>
</table>


Considerations:



* If any argument is `null`, `null` will be returned.
* If search is not found in `original`, `original` will be returned.

<table>
  <tr>
  </tr>
</table>



Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
	RETURN replace('hello', 'l', 'w')
$$) as (str_array agtype);
```


Result:


<table>
  <tr>
   <td>new_str
   </td>
  </tr>
  <tr>
   <td>“Hewwo”
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## split

`split()` returns a list of strings resulting from the splitting of the original string around matches of the given delimiter.

Syntax: `split(original, split_delimiter)`

Returns:


```
An agtype list of agtype strings.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>original
   </td>
   <td>An expression that returns a string.
   </td>
  </tr>
  <tr>
   <td>split_delimiter
   </td>
   <td>The string with which to split original.
   </td>
  </tr>
</table>


Considerations:



* `split(null, splitDelimiter)` and `split(original, null)` both return `null`

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN split('one,two', ',')
$$) as (split_list agtype);
```


Result:


<table>
  <tr>
   <td><strong>split_list</strong>
   </td>
  </tr>
  <tr>
   <td>[“one”,”two”]
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## left

`left()` returns a string containing the specified number of leftmost characters of the original string.

Syntax: `left(original, length)`

Returns:


```
An agtype string.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>original
   </td>
   <td>An expression that returns a string.
   </td>
  </tr>
  <tr>
   <td>n
   </td>
   <td>An expression that returns a positive integer.
   </td>
  </tr>
</table>


Considerations:



* `left(null, length)` and `left(null, null)` both return `null`
* `left(original, null)` will raise an error.
* If `length` is not a positive integer, an error is raised.
* If `length` exceeds the size of `original`, `original` is returned.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
	RETURN left('Hello', 3)
$$) as (new_str agtype);
```


Result:


<table>
  <tr>
   <td>new_str
   </td>
  </tr>
  <tr>
   <td>“Hel”
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## right

`right()` returns a string containing the specified number of rightmost characters of the original string.

Syntax: `right(original, length)`

Returns:


```
An agtype string.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>original
   </td>
   <td>An expression that returns a string.
   </td>
  </tr>
  <tr>
   <td>n
   </td>
   <td>An expression that returns a positive integer.
   </td>
  </tr>
</table>


Considerations:



* `right(null, length)` and `right(null, null)` both return `null`
* `right(original, null)` will raise an error.
* If `length` is not a positive integer, an error is raised.
* If `length` exceeds the size of `original`, `original` is returned.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN right('hello', 3)
$$) as (new_str agtype);
```


Result:


<table>
  <tr>
   <td>new_str
   </td>
  </tr>
  <tr>
   <td>“llo”
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## substring

`substring()` returns a substring of the original string, beginning with a 0-based index start and length.

Syntax: <code>substring(original, <strong>start</strong> [, <strong>length</strong>])</code>

Returns:


```
An agtype string.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>original
   </td>
   <td>An expression that returns a string.
   </td>
  </tr>
  <tr>
   <td>start
   </td>
   <td>An expression denoting the position at which the substring will begin.
   </td>
  </tr>
  <tr>
   <td>length
   </td>
   <td>An optional expression that returns a positive integer, denoting how many characters of the original expression will be returned.
   </td>
  </tr>
</table>


Considerations:



* `start` uses a zero-based index.
* If `length` is omitted, the function returns the substring starting at the position given by `start` and extending to the end of `original`.
* If `original` is `null`, `null` is returned.
* If either `start` or `length` is `null` or a negative integer, an error is raised.
* If `start` is 0, the substring will start at the beginning of `original`.
* If `length` is 0, the empty string will be returned.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN substring('hello', 1, 3), substring('hello', 2)
$$) as (sub_str1 agtype, sub_str2 agtype);
```


Result:


<table>
  <tr>
   <td>sub_str1
   </td>
   <td>sub_str2
   </td>
  </tr>
  <tr>
   <td>‘ell’
   </td>
   <td>‘llo’
   </td>
  </tr>
  <tr>
   <td colspan="2" >1 row(s) returned
   </td>
  </tr>
</table>



## rTrim

`rTrim()` returns the original string with trailing whitespace removed.

Syntax: `rTrim(original)`

Returns:


```
An agtype string.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>original
   </td>
   <td>An expression that returns a string
   </td>
  </tr>
</table>


Considerations:



* `rTrim(null)` returns `null`

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN rTrim(' hello ')
$$) as (right_trimmed_str agtype);
```


Result:


<table>
  <tr>
   <td>right_trimmed_str
   </td>
  </tr>
  <tr>
   <td>" hello"
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## lTrim

`lTrim()` returns the original string with leading whitespace removed.

Syntax: `lTrim(original)`

Returns:


```
An agtype string.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>original
   </td>
   <td>An expression that returns a string
   </td>
  </tr>
</table>


Considerations:



* `lTrim(null)` returns `null`

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN lTrim(' hello ')
$$) as (left_trimmed_str agtype);
```


Result:


<table>
  <tr>
   <td>left_trimmed_str
   </td>
  </tr>
  <tr>
   <td>“hello ”
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## trim

`trim()` returns the original string with leading and trailing whitespace removed.

Syntax: `trim(original)`

Returns:


```
An agtype string.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>original
   </td>
   <td>An expression that returns a string
   </td>
  </tr>
</table>


Considerations:



* `trim(null)` returns `null`

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN trim(' hello ')
$$) as (trimmed_str agtype);
```


Result:


<table>
  <tr>
   <td>trimmed_str
   </td>
  </tr>
  <tr>
   <td>“hello”
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## toLower

`toLower()` returns the original string in lowercase.

Syntax: `toLower(original)`

Returns:


```
An agtype string.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>original
   </td>
   <td>An expression that returns a string
   </td>
  </tr>
</table>


Considerations:



* `toLower(null)` returns `null`

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN toLower('HELLO')
$$) as (lower_str agtype);
```


Result:


<table>
  <tr>
   <td>lower_str
   </td>
  </tr>
  <tr>
   <td>“hello”
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## toUpper

`toUpper() `returns the original string in uppercase.

Syntax: `toUpper(original)`

Returns:


```
An agtype string.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>original
   </td>
   <td>An expression that returns a string
   </td>
  </tr>
</table>


Considerations:



* `toUpper(null)` returns `null`

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN toUpper('hello')
$$) as (upper_str agtype);
```


Result:


<table>
  <tr>
   <td><code>upper_str</code>
   </td>
  </tr>
  <tr>
   <td>“HELLO”
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## reverse

`reverse()` returns a string in which the order of all characters in the original string have been reversed.

Syntax: `reverse(original)`

Returns:


```
An agtype string.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>original
   </td>
   <td>An expression that returns a string
   </td>
  </tr>
</table>


Considerations:



* `reverse(null)` returns `null`.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN reverse("hello")
$$) as (reverse_str agtype);
```


Result:


<table>
  <tr>
   <td>reverse_str
   </td>
  </tr>
  <tr>
   <td>“olleh”
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## toString

`toString()` converts an integer, float or boolean value to a string.

Syntax: `toString(expression)`

Returns:


```
A string.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An expression that returns a number, a boolean, or a string.
   </td>
  </tr>
</table>


Considerations:



* `toString(null)` returns `null`
* If expression is a string, it will be returned unchanged.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN toString(11.5),toString('a string'), toString(true)
$$) as (float_to_str agtype, str_to_str agtype, bool_to_string);
```


Result:


<table>
  <tr>
   <td>float_to_str
   </td>
   <td>str_to_str
   </td>
   <td>bool_to_str
   </td>
  </tr>
  <tr>
   <td>"11.5"
   </td>
   <td>"a string"
   </td>
   <td>"true"
   </td>
  </tr>
  <tr>
   <td colspan="3" >1 row(s) returned
   </td>
  </tr>
</table>


# File: age-website-master/docs/functions/trigonometric_functions.md
# Trigonometric Functions


## degrees

`degrees()` converts radians to degrees.

Syntax: `degrees(expression)`

Returns:


```
An agtype float.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype number expression that represents the angle in radians.
   </td>
  </tr>
</table>


Considerations:



* `degrees(null)` returns `null`.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN degrees(3.14159)
$$) as (deg agtype);
```


The number of degrees in something close to pi is returned.

Results:


<table>
  <tr>
   <td>deg
   </td>
  </tr>
  <tr>
   <td>179.99984796050427
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## radians

`radians()` converts degrees to radians.

Syntax: `radians(expression)`

Returns:


```
An agtype float.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype number expression that represents the angle in degrees.
   </td>
  </tr>
</table>


Considerations:



* `radians(null)` returns `null`.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN radians(180)
$$) as (rad agtype);
```


The number of degrees in something close to pi is returned.

Results:


<table>
  <tr>
   <td>rad
   </td>
  </tr>
  <tr>
   <td>3.14159265358979
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## pi

`pi()` returns the mathematical constant pi.

Syntax: `pi()`

Returns:


```
An agtype float.
```


Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN pi()
$$) as (p agtype);
```


The constant pi is returned.

Result:


<table>
  <tr>
   <td>pi
   </td>
  </tr>
  <tr>
   <td>3.141592653589793
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## sin

`sin()` returns the sine of a number.

Syntax: `sin(expression)`

Returns:


```
An agtype float.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype number expression that represents the angle in radians.
   </td>
  </tr>
</table>


Considerations:



* `sin(null)` returns `null`.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN sin(0.5)
$$) as (s agtype);
```


The sine of 0.5 is returned.

Results:


<table>
  <tr>
   <td>s
   </td>
  </tr>
  <tr>
   <td>0.479425538604203
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## cos

`cos()` returns the cosine of a number.

Syntax: `cos(expression)`

Returns:


```
An agtype float.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype expression that represents the angle in radians.
   </td>
  </tr>
</table>


Considerations:



* `cos(null)` returns `null`.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN cos(0.5)
$$) as (c agtype);
```


The cosine of 0.5 is returned.

Results:


<table>
  <tr>
   <td>c
   </td>
  </tr>
  <tr>
   <td>0.8775825618903728
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## tan

`tan()` returns the tangent of a number.

Syntax: `tan(expression)`

Returns:


```
An agtype float.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype number expression that represents the angle in radians.
   </td>
  </tr>
</table>


Considerations:



* `tan(null)` returns `null`.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN tan(0.5)
$$) as (t agtype);
```


The tangent of 0.5 is returned.

Results:


<table>
  <tr>
   <td>t
   </td>
  </tr>
  <tr>
   <td>0.5463024898437905
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## Cot

`cot()` returns the cotangent of a number.

Syntax: `cot(expression)`

Returns:


```
A float.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype number expression that represents the angle in radians.
   </td>
  </tr>
</table>


Considerations:



* `cot(null)` returns `null`.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN cot(0.5)
$$) as (t agtype);
```


The cotangent of 0.5 is returned.

Results:


<table>
  <tr>
   <td>t
   </td>
  </tr>
  <tr>
   <td>1.830487721712452
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## asin

`asin()` returns the arcsine of a number.

Syntax: `asin(expression)`

Returns:


```
An agtype float.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype number expression that represents the angle in radians.
   </td>
  </tr>
</table>


Considerations:



* `asin(null)` returns `null`.
* If (expression &lt; -1) or (expression > 1), then `asin(expression)` returns `null`.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN asin(0.5)
$$) as (arc_s agtype);
```


The arcsine of 0.5 is returned.

Results:


<table>
  <tr>
   <td>arc_s
   </td>
  </tr>
  <tr>
   <td>0.523598775598299
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## acos

`acos()` returns the arccosine of a number.

Syntax: `acos(expression)`

Returns:


```
An agtype float.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype number expression that represents the angle in radians.
   </td>
  </tr>
</table>


Considerations:



* `acos(null)` returns `null`.
* If (expression &lt; -1) or (expression > 1), then `acos(expression)` returns `null`.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN acos(0.5)
$$) as (arc_c agtype);
```


The arccosine of 0.5 is returned.

Results:


<table>
  <tr>
   <td>arc_c
   </td>
  </tr>
  <tr>
   <td>1.0471975511965979
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## atan

`atan()` returns the arctangent of a number.

Syntax:`atan(expression)`

Returns:


```
An agtype float.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression
   </td>
   <td>An agtype number expression that represents the angle in radians.
   </td>
  </tr>
</table>


Considerations:



* `atan(null)` returns `null`.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN atan(0.5)
$$) as (arc_t agtype);
```


The arctangent of 0.5 is returned.

Results:


<table>
  <tr>
   <td>arc_t
   </td>
  </tr>
  <tr>
   <td>0.463647609000806
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



## atan2

`atan2()` returns the arctangent2 of a set of coordinates in radians.

Syntax: `atan2(expression1, expression2)`

Returns:


```
An agtype float.
```


Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>expression1
   </td>
   <td>An agtype number expression for y that represents the angle in radians.
   </td>
  </tr>
  <tr>
   <td>expression2
   </td>
   <td>An agtype number expression for x that represents the angle in radians.
   </td>
  </tr>
</table>


Considerations:



* `atan2(null, null)`, `atan2(null, expression2)` and `atan(expression1, null)` all return null.

Query:


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN atan2(0.5, 0.6)
$$) as (arc_t2 agtype);
```


The arctangent2 of 0.5 and 0.6 is returned.

Results:


<table>
  <tr>
   <td>arc_t2
   </td>
  </tr>
  <tr>
   <td>0.694738276196703
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>



# File: age-website-master/docs/functions/user_functions.md
# User Defined Functions

Users may add custom functions to AGE. When using Cypher functions, all function calls with a Cypher query use the default namespace of: `ag_catalog`. However if a user wants to use a function outside of this namespace, they may do so by adding the namespace before the function name.

Syntax: `namespace_name.function_name`

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
RETURN pg_catalog.sqrt(25)
$$) as (result agtype);
```


Result:


<table>
  <tr>
   <td>result
   </td>
  </tr>
  <tr>
   <td>25
   </td>
  </tr>
  <tr>
   <td>1 row(s) returned
   </td>
  </tr>
</table>




# File: age-website-master/docs/intro/aggregation.md
# Aggregation


## Introduction 

Generally an aggregation `aggr(expr)` processes all matching rows for each aggregation key found in an incoming record (keys are compared using [equivalence](../intro/comparability.md)).

In a regular aggregation (i.e. of the form `aggr(expr)`), the list of aggregated values is the list of candidate values with all null values removed from it.

## Data Setup

```postgresql
SELECT * FROM cypher('graph_name', $$
	CREATE (a:Person {name: 'A', age: 13}),
	(b:Person {name: 'B', age: 33, eyes: "blue"}),
	(c:Person {name: 'C', age: 44, eyes: "blue"}),
	(d1:Person {name: 'D', eyes: "brown"}),
	(d2:Person {name: 'D'}),
	(a)-[:KNOWS]->(b),
	(a)-[:KNOWS]->(c),
	(a)-[:KNOWS]->(d1),
	(b)-[:KNOWS]->(d2),
	(c)-[:KNOWS]->(d2)
$$) as (a agtype);
```

## Auto Group By
To calculate aggregated data, Cypher offers aggregation, analogous to SQL’s `GROUP BY`.

Aggregating functions take a  set of values and calculate An aggregated value over them. Examples are [`avg()`](../functions/aggregate_functions.md#avg) that calculates the average of multiple numeric values, or [`min()`](../functions/aggregate_functions.md#min) that finds the smallest numeric or string value in a set of values. When we say below that an aggregating function operates on a set of values, we mean these to be the result of the application of the inner expression(such as `n.age`) to all the records within the same aggregation group.

Aggregation can be computed over all the matching subgraphs, or it can be further divided by introducing grouping keys. These are non-aggregate expressions, that are used to group the values going into the aggregate functions.

Assume we have the following return statement:
```postgresql
SELECT * FROM cypher('graph_name', $$
	MATCH (v:Person)
	RETURN v.name, count(*)
$$) as (grouping_key agtype, count agtype);
```

<table>
  <tr>
   <td>count</td>
   <td>key</td>
  </tr>
  <tr>
   <td>"A"</td>
   <td>1</td>
  </tr>
  <tr>
   <td>"B"</td>
   <td>1</td>
  </tr>
  <tr>
   <td>"C"</td>
   <td>1</td>
  </tr>
  <tr>
   <td>"D"</td>
   <td>2</td>
  </tr>
  <tr>
   <td colspan="2">1 row</td>
  </tr>
</table>


We have two return expressions: `grouping_key`, and `count(*)`. The first, `grouping_key`, is not an aggregate function, and so it will  be  the  grouping  key. The latter, `count(*)` is an aggregate expression. The matching subgraphs will be divided into different  buckets, depending on the grouping key. The aggregate function will then be run on these buckets, calculating an aggregate value per bucket. 

## Sorting on aggregate functions

To use aggregations to sort the result set, the aggregation must be included in the `RETURN` to be used in the `ORDER BY`.

```postgresql
SELECT *
FROM cypher('graph_name', $$
	MATCH (me:Person)-[]->(friend:Person)
	RETURN count(friend), me
	ORDER BY count(friend)
$$) as (friends agtype, me agtype);
```

## Distinct aggregation
In a distinct aggregation (i.e. of the form `aggr(DISTINCT expr)`), the list of aggregated values is the list of candidate values with all null values  removed from it. Furthermore, in a distinct aggregation, only one of all equivalent candidate values is included in the list of aggregated values, i.e. duplicates under equivalence are  removed. 


The `DISTINCT` operator works in conjunction with aggregation. It is used to make all values unique before running them  through an aggregate function.

```postgresql
SELECT *
FROM cypher('graph_name', $$
	MATCH (v:Person)
	RETURN count(DISTINCT v.eyes), count(v.eyes)
$$) as (distinct_eyes agtype, eyes agtype);
```

<table>
  <tr>
   <td>distinct_eyes</td>
   <td>eyes</td>
  </tr>
  <tr>
   <td>2</td>
   <td>3</td>
  </tr>
  <tr>
   <td colspan="2">1 row</td>
  </tr>
</table>

## Ambiguous Grouping Statements

This feature of not requiring the user to specify their grouping keys for a query allows for ambiguity on what Cypher should qualify as their grouping keys. For more details [click here.](https://opencypher.org/articles/2017/07/27/ocig1-aggregations-article/)

Data Setup 
```postgresql
SELECT * FROM cypher('graph_name', $$
CREATE (:L {a: 1, b: 2, c: 3}),
       (:L {a: 2, b: 3, c: 1}),
       (:L {a: 3, b: 1, c: 2})
$$) as (a agtype);
```

### Invalid Query in AGE
AGE's solution to this problem is to not allow a `WITH` or `RETURN` column to combine aggregate functions with variables that are not explicitly listed in another column of the same `WITH` or `RETURN` clause.



Query:
```postgresql
SELECT * FROM cypher('graph_name', $$
	MATCH (x:L)
	RETURN x.a + count(*) + x.b + count(*) + x.c
$$) as (a agtype);
```

Result:
```postgresql
ERROR:  "x" must be either part of an explicitly listed key or used inside an aggregate function
LINE 3: RETURN x.a + count(*) + x.b + count(*) + x.c
```


### Valid Query in AGE
Columns that do not include an aggregate function in AGE are considered to be the grouping keys for that `WITH` or `RETURN` clause. 

For the above query, the user could rewrite the query is several ways that will return results

Query:
```postgresql
SELECT * FROM cypher('graph_name', $$
	MATCH (x:L)
	RETURN (x.a + x.b + x.c) + count(*) + count(*), x.a + x.b + x.c
$$) as (count agtype, key agtype);
```

`x.a + x.b + x.c` is the grouping key. Grouping keys created like this must include parenthesis.

Results
<table>
  <tr>
   <td>count</td>
   <td>key</td>
  </tr>
  <tr>
   <td>12</td>
   <td>6</td>
  </tr>
  <tr>
   <td colspan="2">1 row</td>
  </tr>
</table>



Query
```postgresql
SELECT * FROM cypher('graph_name', $$
	MATCH (x:L)
	RETURN x.a + count(*) + x.b + count(*) + x.c, x.a, x.b, x.c
$$) as (count agtype, a agtype, b agtype, c agtype);
```

`x.a`, `x.b`, and `x.c` will be considered different grouping keys

Results:

<table>
  <thead>
  <tr>
   <td>count</td>
   </td>a<td>
   </td>b<td>
   </td>c<td>
  </tr>
  </thead>
  <tr>
   <td>8</td>
   <td>3</td>
   <td>1</td>
   <td>2</td>
  </tr>
  <tr>
   <td>8</td>
   <td>2</td>
   <td>3</td>
   <td>1</td>
  </tr>
  <tr>
   <td>8</td>
   <td>1</td>
   <td>2</td>
   <td>3</td>
  </tr>
  <tr>
   <td colspan="4">3 rows</td>
  </tr>
</table>

### Vertices and edges in ambiguous grouping

Alternatively, the grouping key can be a vertex or edge, and then any properties of the vertex or edge can be specified without being explicitly stated in a `WITH` or `RETURN` column.

```postgresql
SELECT * FROM cypher('graph_name', $$
	MATCH (x:L)
	RETURN count(*) + count(*) + x.a + x.b + x.c, x
$$) as (count agtype, key agtype);
```

Results will be grouped on `x`, because it is safe to assume that properties be considered unecessary for grouping to be unambiguous.

Results
<table>
  <thead>
  <tr>
   <td>count</td>
   </td>key<td>
  </tr>
  </thead>
  <tr>
   <td>8</td>
   <td>{"id": 1407374883553283, "label": "L", "properties": {"a": 3, "b": 1, "c": 2}}::vertex</td>
  </tr>
  <tr>
   <td>8</td>
   <td>{"id": 1407374883553281, "label": "L", "properties": {"a": 1, "b": 2, "c": 3}}::vertex</td>
  </tr>
  <tr>
   <td>8</td>
   <td>{"id": 1407374883553282, "label": "L", "properties": {"a": 2, "b": 3, "c": 1}}::vertex</td>
  </tr>
  <tr>
   <td colspan="4">3 rows</td>
  </tr>
</table>


### Hiding unwanted grouping keys

If the grouping key is considered unecessary for the query output, the aggregation can be done in a `WITH` clause then passing information to the `RETURN` clause.

```postgresql
SELECT * FROM cypher('graph_name', $$
	MATCH (x:L)
	WITH count(*) + count(*) + x.a + x.b + x.c as column, x
	RETURN column
$$) as (a agtype);
```

Results
<table>
  <thead>
  <tr>
   <td>a</td>
  </tr>
  </thead>
  <tr>
   <td>8</td>
  </tr>
  <tr>
   <td>8</td>
  </tr>
  <tr>
   <td>8</td>
  </tr>
  <tr>
   <td colspan="1">3 rows</td>
  </tr>
</table>






















# File: age-website-master/docs/intro/agload.md
# Importing Graph from Files 
You can use the following instructions to create a graph from the files. This document explains 
- information about the current branch that includes the functions to load graphs from files
- explanation of the functions that enable the creation of graphs from files 
- the structure of CSV files that load functions as input, do and do not. 
- A simple source code example to load countries and cities from the files. 


User can load graph in two steps 
- Load Vertices in the first step
- Load Edges in the second step

**User must create graph and labels before loading data from files**

## Load Graph functions 
Following are the details about the functions to create vertices and edges from the file. 

Function `load_labels_from_file` is used to load vertices from the CSV files. 

```postgresql
load_labels_from_file('<graph name>', 
                      '<label name>',
                      '<file path>')
```

By adding the fourth parameter user can exclude the id field. *** Use this when there is no id field in the file***

```postgresql
load_labels_from_file('<graph name>', 
                      '<label name>',
                      '<file path>', 
                      false)
```

Function `load_edges_from_file` can be used to load edges from the CSV file. Please see the file structure in the following. 

Note: make sure that ids in the edge file are identical to ones that are in vertices files. 

```postgresql
load_edges_from_file('<graph name>',
                    '<label name>',
                    '<file path>');
```

## Explanation about the CSV format
Following is the explanation about the structure for CSV files for vertices and edges.

- A CSV file for nodes shall be formatted as following; 

| field name | Field description                                            |
| ---------- | ------------------------------------------------------------ |
| id         | it shall be the first column of the file and all values shall be a positive integer. <br>This is an optional field when `id_field_exists` is ***false***. <br>However, it should be present when `id_field_exists` is ***not*** set to false.  |
| Properties | all other columns contains the properties for the nodes. <br>Header row shall contain the name of property |

- Similarly, a CSV file for edges shall be formatted as follows 

| field name        | Field description                                            |
| ----------------- | ------------------------------------------------------------ |
| start_id          | node id of the node from where the edge is stated. <br>This id shall be present in nodes.csv file. |
| start_vertex_type | class of the node                                            |
| end_id            | end id of the node at which the edge shall be terminated    |
| end_vertex_type   | Class of the node                                            |
| properties        | properties of the edge. the header shall contain the property name |

Example files can be viewed at `regress/age_load/data`

## Example SQL script 

- Load and create graph 
```postgresql
LOAD 'age';

SET search_path TO ag_catalog;
SELECT create_graph('agload_test_graph');
```

- Create label `Country` and load vertices from csv file. *** Note this CSV file has id field ***

```postgresql
SELECT create_vlabel('agload_test_graph','Country');
SELECT load_labels_from_file('agload_test_graph',
                             'Country',
                             'age/regress/age_load/data/countries.csv');
```

- Create label `City` and load vertices from csv file. *** Note this CSV file has id field ***

```postgresql
SELECT create_vlabel('agload_test_graph','City');
SELECT load_labels_from_file('agload_test_graph',
                             'City', 
                             'age/regress/age_load/data/cities.csv');
```

- Create label `has_city` and load edges from csv file.

```postgresql
SELECT create_elabel('agload_test_graph','has_city');
SELECT load_edges_from_file('agload_test_graph', 'has_city',
     'age/regress/age_load/data/edges.csv');
```

- Check if the graph has been loaded properly

```postgresql
SELECT table_catalog, table_schema, table_name, table_type
FROM information_schema.tables
WHERE table_schema = 'agload_test_graph';

SELECT COUNT(*) FROM agload_test_graph."Country";
SELECT COUNT(*) FROM agload_test_graph."City";
SELECT COUNT(*) FROM agload_test_graph."has_city";

SELECT COUNT(*) FROM cypher('agload_test_graph', $$MATCH(n) RETURN n$$) as (n agtype);
SELECT COUNT(*) FROM cypher('agload_test_graph', $$MATCH (a)-[e]->(b) RETURN e$$) as (n agtype);
```

### Creating vertices without id field in the file. 

- Create label `Country2` and load vertices from csv file. *** Note this CSV file has no id field ***

```postgresql
SELECT create_vlabel('agload_test_graph','Country2');
SELECT load_labels_from_file('agload_test_graph',
                             'Country2',
                             'age/regress/age_load/data/countries.csv', 
                             false);
```

- Create label `City2` and load vertices from csv file. *** Note this CSV file has no id field ***
```postgresql
SELECT create_vlabel('agload_test_graph','City2');
SELECT load_labels_from_file('agload_test_graph',
                             'City2',
                             'age/regress/age_load/data/cities.csv', 
                             false);
```
- Check if the graph has been loaded properly and perform difference analysis between ids created automatically and picked from the files.

- Labels `Country` and `City` were created with id field in the file
- Labels `Country2` and `City2` were created with no id field in the file. 
```postgresql
SELECT COUNT(*) FROM agload_test_graph."Country2";
SELECT COUNT(*) FROM agload_test_graph."City2";

SELECT id FROM agload_test_graph."Country" LIMIT 10;
SELECT id FROM agload_test_graph."Country2" LIMIT 10;

SELECT * FROM cypher('agload_test_graph', $$MATCH(n:Country {iso2 : 'BE'})
    RETURN id(n), n.name, n.iso2 $$) as ("id(n)" agtype, "n.name" agtype, "n.iso2" agtype);
SELECT * FROM cypher('agload_test_graph', $$MATCH(n:Country2 {iso2 : 'BE'})
    RETURN id(n), n.name, n.iso2 $$) as ("id(n)" agtype, "n.name" agtype, "n.iso2" agtype);

SELECT * FROM cypher('agload_test_graph', $$MATCH(n:Country {iso2 : 'AT'})
    RETURN id(n), n.name, n.iso2 $$) as ("id(n)" agtype, "n.name" agtype, "n.iso2" agtype);
SELECT * FROM cypher('agload_test_graph', $$MATCH(n:Country2 {iso2 : 'AT'})
    RETURN id(n), n.name, n.iso2 $$) as ("id(n)" agtype, "n.name" agtype, "n.iso2" agtype);

SELECT drop_graph('agload_test_graph', true);
```


# File: age-website-master/docs/intro/comparability.md
# Comparability, Equality, Orderability and Equivalence

AGE already has good semantics for equality within the primitive types (booleans, strings,integers, and floats) and maps. Furthermore, Cypher has good semantics for comparability and orderability for integers, floats, and strings, within each of the types. However, working with values of different types deviates from Postgres’ defined logic and the openCypher specification:



* Comparability between values of different types is defined. This deviation is particularly pronounced when it occurs as part of the evaluation of predicates (in WHERE).
* ORDER BY will not fail if the values passed to it have different types.

The underlying conceptual model is complex and sometimes inconsistent. This leads to an unclear relationship between comparison operators, equality, grouping, and ORDER BY:
* Comparability and orderability are aligned with each other consistently, as all types can be ordered and compared.
* The difference between equality and equivalence, as exposed by `IN`, `=`, `DISTINCT`, and grouping, in AGE is limited to testing two instances of the value null to each other
    * In equality, `null = null` is `null`.
    * In equivalence, used by `DISTINCT` and when grouping values, two null values are always treated as being the same value.
    * However, equality treats null values differently if they are an element of a list or a map value.

## Concepts

The openCypher specification features four distinct concepts related to equality and ordering:


### Comparability

Comparability is used by the inequality operators (>, &lt;, >=, &lt;=), and defines the underlying semantics of how to compare two values.


### Equality

Equality is used by the equality operators (=, &lt;>), and the list membership operator (`IN`). It defines the underlying semantics to determine if two values are the same in these contexts. Equality is also used implicitly by literal maps in node and relationship patterns, since such literal maps are merely a shorthand notation for equality predicates.


### Orderability

Orderability is used by the `ORDER BY` clause, and defines the underlying semantics of how to order values.


### Equivalence

Equivalence is used by the `DISTINCT` modifier and by grouping in projection clauses (`WITH`, `RETURN`), and defines the underlying semantics to determine if two values are the same in these contexts.

## Comparability and equality

Comparison operators need to function as one would expect comparison operators to function - equality and comparability. But, at the same time, they need to allow the sorting of column data - equivalence and orderability.

Unfortunately, it may not be possible to implement separate comparison operators for equality and comparison operations, and, equivalence and orderability operations, in PostgreSQL, for the same query. So we prioritize equivalence and orderability over equality and comparability to allow for ordering of output data.


### Comparability

Comparability is defined between any pair of values, as specified below.

* Numbers 
    * Numbers of different types (excluding NaN values and the Infinities) are compared to each other as if both numbers would have been coerced to arbitrary precision big decimals(currently outside the Cypher type system) before comparing them with each other numerically in ascending order.
    * Comparison to any value that is not also Number follows the rules of orderability.
    * Floats don’t have the required precision to represent all of the whole numbers in the range of agtype integer and agtype numeric. When casting an integer or numeric agtype to a float, unexpected results can occur when casting values in the high and low range
    * Integers
        * Integers are compared numerically in ascending order.
    * Floats
        * Floats (excluding NaN values and the Infinities) are compared numerically in ascending order.
        * Positive infinity is of type `FLOAT`, equal to itself and greater than any other number, except NaN values.
        * Negative infinity is of type `FLOAT`, equal to itself and less than any other number.
        * NaN values are comparable to each and greater than any other float value.
    * Numeric
        * Numerics are compared numerically in ascending order.
* Booleans
    * Booleans are compared such that false is less than true.
    * Comparison to any value that is not also a boolean follows the rules of orderability.
* Strings
    * Strings are compared in dictionary order, i.e. characters are compared pairwise in ascending order from the start of the string to the end. Characters missing in a shorter string are considered to be less than any other character. For example, `'a' < 'aa'`.
    * Comparison to any value that is not also a string follows the rules of orderability.
* Lists
    * Lists are compared in sequential order, i.e. list elements are compared pairwise in ascending order from the start of the list to the end. Elements missing in a shorter list are considered to be less than any other value (including null values). For example, `[1] < [1, 0]` but also `[1] < [1, null]`.
    * Comparison to any value that is not also a list follows the rules of orderability.
* Maps
    * The comparison order for maps is unspecified and left to implementations.
    * The comparison order for maps must align with the equality semantics outlined below. In consequence, any map that contains an entry that maps its key to a null value is incomparable. For example, `{a: 1} <= {a: 1, b: null}` evaluates to null.
    * Comparison to any value that is not also a regular map follows the rules of orderability.

Entities
* Vertices
    * The comparison order for vertices is based on the assigned graphid.
* Edges
    * The comparison order for edges is based on the assigned graphid.
* Paths
    * Paths are compared as if they were a list of alternating nodes and relationships of the path from the start node to the end node. For example, given nodes `n1`, `n2`, `n3`, and relationships `r1` and `r2`, and given that `n1 < n2 < n3` and `r1 < r2`, then the path `p1` from `n1` to `n3` via `r1` would be less than the path `p2` to `n1` from `n2` via `r2`. 
    * Expressed in terms of lists: 
```
p1 < p2
<=> [n1, r1, n3] < [n1, r2, n2]
<=> n1 < n1 || (n1 = n1 && [r1, n3] < [r2, n2])
<=> false || (true && [r1, n3] < [r2, n2])<=> [r1, n3] < [r2, n2]
<=> r1 < r2 || (r1 = r2 && n3 < n2)
<=> true || (false && false)
<=> true
```
    * Comparison to any value that is not also a path will return false.
* NULL
    * null is incomparable with any other value (including other null values.)


## Orderability Between different Agtypes

The ordering of different Agtype, when using &lt;, &lt;=, >, >= from smallest value to largest value is: 

1. Path
2. Edge
3. Vertex
4. Object
5. Array
6. String
7. Bool
8. Numeric, Integer, Float
9. NULL

Note: This is subject to change in future releases.




# File: age-website-master/docs/intro/cypher.md
# The AGE Cypher Query Format

Cypher queries are constructed using a function called cypher in ag_catalog which returns a Postgres SETOF [records](https://www.postgresql.org/docs/11/xfunc-sql.html#XFUNC-SQL-FUNCTIONS-RETURNING-SET).


## Cypher()

`cypher()` executes the cypher query passed as an argument.

Syntax `cypher(graph_name, query_string, parameters)`

Returns


```
A SETOF records
```


Arguments:


<table>
  <tr>
   <td>Argument Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>graph_name
   </td>
   <td>The target graph for the Cypher query.
   </td>
  </tr>
  <tr>
   <td>query_string
   </td>
   <td>The Cypher query to be executed.
   </td>
  </tr>
  <tr>
   <td>parameters
   </td>
   <td>An optional map of parameters used for Prepared Statements. Default is NULL. 
   </td>
  </tr>
</table>


Considerations:
* If a Cypher query does not return results, a record definition still needs to be defined. 
* The parameter map can only be used with [Prepared Statements](../advanced/prepared_statements). An error will be thrown otherwise.

Query:


```postgresql
SELECT * FROM cypher('graph_name', $$ 
/* Cypher Query Here */ 
$$) AS (result1 agtype, result2 agtype);
```

## Cypher in an Expression

Cypher may not be used as part of an expression, use a subquery instead. See [Advanced Cypher Queries](../advanced/advanced.md#cypher-in-sql-expressions) for information about how to use Cypher queries with Expressions


## SELECT Clause

Calling Cypher in the `SELECT` clause as an independent column is not allowed. However Cypher may be used when it belongs as a conditional. 

Not Allowed:


```postgresql
SELECT 
    cypher('graph_name', $$
         MATCH (v:Person)
         RETURN v.name
     $$);
```



```
ERROR:  cypher(...) in expressions is not supported
LINE 3: 	cypher('graph_name', $$
        	^
HINT:  Use subquery instead if possible.
```



# File: age-website-master/docs/intro/graphs.md
# Graphs

A graph consists of a set of vertices and edges, where each individual node and edge possesses a map of properties. A vertex is the basic object of a graph, that can exist independently of everything else in the graph. An edge creates a directed connection between two vertices.


## Create a Graph

To create a graph, use the `create_graph` function, located in the `ag_catalog` namespace.


### create_graph()

Syntax: `create_graph(graph_name);`

Returns:

```
void
```

Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>graph_name
   </td>
   <td>Name of the graph to be created
   </td>
  </tr>
</table>


Considerations



* This function will not return any results. The graph is created if there is no error message.
* Tables needed to set up the graph are automatically created.

Example:

```sql
SELECT * FROM ag_catalog.create_graph('graph_name');
```

## Delete a Graph

To delete a graph, use the `drop_graph` function, located in the `ag_catalog` namespace.


### drop_graph()

Syntax: `drop_graph(graph_name, cascade);`

Returns:

```
void
```

Arguments:


<table>
  <tr>
   <td>Name
   </td>
   <td>Description
   </td>
  </tr>
  <tr>
   <td>graph_name
   </td>
   <td>Name of the graph to be deleted
   </td>
  </tr>
  <tr>
   <td>cascade
   </td>
   <td>A boolean that will will delete labels and data that depend on the graph.
   </td>
  </tr>
</table>


Considerations:



* This function will not return any results. If there is no error message the graph has been deleted.
* It is recommended to set the cascade option to true, otherwise everything in the graph must be manually dropped with SQL DDL commands.

Example:

```sql
SELECT * FROM ag_catalog.drop_graph('graph_name', true);
```

## How Graphs Are Stored In Postgres

When creating graphs with AGE, a Postgres namespace will be generated for every individual graph. 
The name and namespace of the created graphs can be seen within the `ag_graph` table from the `ag_catalog` namespace:
```sql
SELECT create_graph('new_graph');

NOTICE:  graph "new_graph" has been created
 create_graph 
--------------

(1 row)

SELECT * FROM ag_catalog.ag_graph;

   name    | namespace 
-----------+-----------
 new_graph | new_graph
(1 row)
```

After creating the graph, two tables are going to be created under the graph's namespace to store vertices and edges: `_ag_label_vertex` and `_ag_label_edge`.
These will be the parent tables of any new vertex or edge label. The query below shows how to retrieve the edge and vertex labels for all the graphs in the database.

```sql
-- Before creating a new vertex label.
SELECT * FROM ag_catalog.ag_label;

       name       | graph | id | kind |          relation          |        seq_name         
------------------+-------+----+------+----------------------------+-------------------------
 _ag_label_vertex | 68484 |  1 | v    | new_graph._ag_label_vertex | _ag_label_vertex_id_seq
 _ag_label_edge   | 68484 |  2 | e    | new_graph._ag_label_edge   | _ag_label_edge_id_seq
(2 rows)

-- Creating a new vertex label.
SELECT create_vlabel('new_graph', 'Person');
NOTICE:  VLabel "Person" has been created
 create_vlabel 
---------------
 
(1 row)

-- After creating a new vertex label.
SELECT * FROM ag_catalog.ag_label;
       name       | graph | id | kind |          relation          |        seq_name         
------------------+-------+----+------+----------------------------+-------------------------
 _ag_label_vertex | 68484 |  1 | v    | new_graph._ag_label_vertex | _ag_label_vertex_id_seq
 _ag_label_edge   | 68484 |  2 | e    | new_graph._ag_label_edge   | _ag_label_edge_id_seq
 Person           | 68484 |  3 | v    | new_graph."Person"         | Person_id_seq
(3 rows)

```

Whenever a vertex label is created with the `create_vlabel()` function, a new table is generated within the `new_graph`'s namespace: `new_graph."<label>"`.
The same works for the `create_elabel()` function for the creation of edge labels. Creating vertices and edges with Cypher will automatically make these tables.

```sql
-- Creating two vertices and one edge.
SELECT * FROM cypher('new_graph', $$
CREATE (:Person {name: 'Daedalus'})-[:FATHER_OF]->(:Person {name: 'Icarus'})
$$) AS (a agtype);
 a 
---
(0 rows)

-- Showing the newly created tables.
SELECT * FROM ag_catalog.ag_label;
       name       | graph | id | kind |          relation          |        seq_name         
------------------+-------+----+------+----------------------------+-------------------------
 _ag_label_vertex | 68484 |  1 | v    | new_graph._ag_label_vertex | _ag_label_vertex_id_seq
 _ag_label_edge   | 68484 |  2 | e    | new_graph._ag_label_edge   | _ag_label_edge_id_seq
 Person           | 68484 |  3 | v    | new_graph."Person"         | Person_id_seq
 FATHER_OF        | 68484 |  4 | e    | new_graph."FATHER_OF"      | FATHER_OF_id_seq
(4 rows)
```

_Note: It is recommended that no DML or DDL commands are executed in the namespace that is reserved for the graph._ 
<!-- Needs clarification. Since search path is set as ag_catalog first in the searh path, all DML and DDL will happen in the ag_catalog namespace. Also should we say schema rather than namespace? 
-->



# File: age-website-master/docs/intro/operators.md
# Operators

## String Specific Comparison Operators

### Data Setup

```postgresql
SELECT * FROM cypher('graph_name', $$
CREATE (:Person {name: 'John'}),
       (:Person {name: 'Jeff'}),
       (:Person {name: 'Joan'}),
       (:Person {name: 'Bill'})
$$) AS (result agtype);
```

### Starts With

Performs case-sensitive prefix searching on strings.

```postgresql
SELECT * FROM cypher('graph_name', $$
	MATCH (v:Person)
	WHERE v.name STARTS WITH "J"
	RETURN v.name
$$) AS (names agtype);
```

Results
<table>
  <thead>
  <tr>
   <td>names</td>
  </tr>
  </thead>
  <tr>
   <td>"John"</td>
  </tr>
  <tr>
   <td>"Jeff"</td>
  </tr>
  <tr>
   <td>"Joan"</td>
  </tr>
  <tr>
   <td colspan="1">3 rows</td>
  </tr>
</table>

### Contains

Performs case-sensitive inclusion searching in strings.

```postgresql
SELECT * FROM cypher('graph_name', $$
	MATCH (v:Person)
	WHERE v.name CONTAINS "o"
	RETURN v.name
$$) AS (names agtype);
```

Results
<table>
  <thead>
  <tr>
   <td>names</td>
  </tr>
  </thead>
  <tr>
   <td>"John</td>
  </tr>
  <tr>
   <td>"Joan</td>
  </tr>
  <tr>
   <td colspan="1">2 rows</td>
  </tr>
</table>


### Ends With

Performs case-sensitive suffix searching on strings.

```postgresql
SELECT * FROM cypher('graph_name', $$
	MATCH (v:Person)
	WHERE v.name ENDS WITH "n"
	RETURN v.name
$$) AS (names agtype);
```

Results
<table>
  <thead>
  <tr>
   <td>names</td>
  </tr>
  </thead>
  <tr>
   <td>"John"</td>
  </tr>
  <tr>
   <td>"Joan"</td>
  </tr>
  <tr>
   <td colspan="1">2 rows</td>
  </tr>
</table>

### Regular Expressions

AGE supports the use of [POSIX regular expressions](https://www.postgresql.org/docs/11/functions-matching.html#FUNCTIONS-POSIX-REGEXP) using the `=~` operator. By default `=~` is case sensitve.


#### Basic String Matching

The `=~` operator when no special characters are given, act like the `=` operator.

```postgresql
SELECT * FROM cypher('graph_name', $$
	MATCH (v:Person)
	WHERE v.name =~ 'John'
	RETURN v.name
$$) AS (names agtype);
```

Results
<table>
  <thead>
  <tr>
   <td>names</td>
  </tr>
  </thead>
  <tr>
   <td>"John"</td>
  </tr>
  <tr>
   <td colspan="1">1 rows</td>
  </tr>
</table>

#### Case insensitive search

Adding `(?i)` at the beginning of the string will make the comparison case insensitive

```postgresql
SELECT * FROM cypher('graph_name', $$
	MATCH (v:Person)
	WHERE v.name =~ '(?i)JoHn'
	RETURN v.name
$$) AS (names agtype);
```

<table>
  <thead>
  <tr>
   <td>names</td>
  </tr>
  </thead>
  <tr>
   <td>"John"</td>
  </tr>
  <tr>
   <td colspan="1">1 rows</td>
  </tr>
</table>


#### The . Wildcard

The . operator acts as a wildcard to match any single character.

```postgresql
SELECT * FROM cypher('graph_name', $$
	MATCH (v:Person)
	WHERE v.name =~ 'Jo.n'
	RETURN v.name
$$) AS (names agtype);
```

<table>
  <thead>
  <tr>
   <td>names</td>
  </tr>
  </thead>
  <tr>
   <td>"John"</td>
  </tr>
  <tr>
   <td>"Joan"</td>
  </tr>
  <tr>
   <td colspan="1">2 rows</td>
  </tr>
</table>

#### The * Wildcard

The * wildcard after a character will match to 0 or more of the previous character

```postgresql
SELECT * FROM cypher('graph_name', $$
	MATCH (v:Person)
	WHERE v.name =~ 'Johz*n'
	RETURN v.name
$$) AS (names agtype);
```

<table>
  <thead>
  <tr>
   <td>names</td>
  </tr>
  </thead>
  <tr>
   <td>"John"</td>
  </tr>
  <tr>
   <td colspan="1">1 rows</td>
  </tr>
</table>


#### The + Operator

The + operator matches to 1 or more the previous character.

```postgresql
SELECT * FROM cypher('graph_name', $$
	MATCH (v:Person)
	WHERE v.name =~ 'Bil+'
	RETURN v.name
$$) AS (names agtype);
```

Results
<table>
  <thead>
  <tr>
   <td>names</td>
  </tr>
  </thead>
  <tr>
   <td>"Bill"</td>
  </tr>
  <tr>
   <td colspan="1">1 row</td>
  </tr>
</table>

#### The . and * wildcards together

You can use the . and * wildcards together to represent the rest of a string.

```postgresql
SELECT * FROM cypher('graph_name', $$
	MATCH (v:Person)
	WHERE v.name =~ 'J.*'
	RETURN v.name
$$) AS (names agtype);
```

<table>
  <thead>
  <tr>
   <td>names</td>
  </tr>
  </thead>
  <tr>
   <td>"John"</td>
  </tr>
  <tr>
   <td>"Jeff"</td>
  </tr>
  <tr>
   <td>"Joan"</td>
  </tr>
  <tr>
   <td colspan="1">2 rows</td>
  </tr>
</table>


## Operator Precedence

Operator precedence in AGE is shown below:


<table>
  <tr>
   <td>Precedence
   </td>
   <td>Operator
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>1
   </td>
   <td>.
   </td>
   <td>Property Access
   </td>
  </tr>
  <tr>
   <td rowspan="2" >2
   </td>
   <td>[]
   </td>
   <td>Map and List Subscripting
   </td>
  </tr>
  <tr>
   <td>()
   </td>
   <td>Function Call
   </td>
  </tr>
  <tr>
   <td rowspan="4" >3
   </td>
   <td>STARTS WITH
   </td>
   <td>Case-sensitive prefix searching on strings
   </td>
  </tr>
  <tr>
   <td>ENDS WITH
   </td>
   <td>Case-sensitive suffix searching on strings
   </td>
  </tr>
  <tr>
   <td>CONTAINS
   </td>
   <td>Case-sensitive inclusion searching on strings
   </td>
  </tr>
  <tr>
   <td>=~
   </td>
   <td>Regular expression string matching
   </td>
  </tr>
  <tr>
   <td>4
   </td>
   <td>-
   </td>
   <td>Unary Minus
   </td>
  </tr>
  <tr>
   <td rowspan="3" >5
   </td>
   <td>IN
   </td>
   <td>Checking if an element exists in a list
   </td>
  </tr>
  <tr>
   <td>IS NULL
   </td>
   <td>Checking a value is NULL
   </td>
  </tr>
  <tr>
   <td>IS NOT NULL
   </td>
   <td>Checking a value is not NULL
   </td>
  </tr>
  <tr>
   <td>6
   </td>
   <td>^
   </td>
   <td>Exponentiation
   </td>
  </tr>
  <tr>
   <td>7
   </td>
   <td>* / %
   </td>
   <td>Multiplication, division and remainder
   </td>
  </tr>
  <tr>
   <td>8
   </td>
   <td>+ -
   </td>
   <td>Addition and Subtraction
   </td>
  </tr>
  <tr>
   <td rowspan="3" >9
   </td>
   <td>= &lt;>
   </td>
   <td>For relational = and ≠ respectively
   </td>
  </tr>
  <tr>
   <td>&lt; &lt;=
   </td>
   <td>For relational &lt; and ≤ respectively
   </td>
  </tr>
  <tr>
   <td>> >=
   </td>
   <td>For relational > and ≥ respectively
   </td>
  </tr>
  <tr>
   <td>10
   </td>
   <td>NOT
   </td>
   <td>Logical NOT
   </td>
  </tr>
  <tr>
   <td>11
   </td>
   <td>AND
   </td>
   <td>Logical AND
   </td>
  </tr>
  <tr>
   <td>12
   </td>
   <td>OR
   </td>
   <td>Logical OR
   </td>
  </tr>
</table>





# File: age-website-master/docs/intro/precedence.md
# Operator Precedence

Operator precedence in AGE is shown below:


<table>
  <tr>
   <td>Precedence
   </td>
   <td>Operator
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>1
   </td>
   <td>.
   </td>
   <td>Property Access
   </td>
  </tr>
  <tr>
   <td rowspan="2" >2
   </td>
   <td>[]
   </td>
   <td>Map and List Subscripting
   </td>
  </tr>
  <tr>
   <td>()
   </td>
   <td>Function Call
   </td>
  </tr>
  <tr>
   <td rowspan="4" >3
   </td>
   <td>STARTS WITH
   </td>
   <td>Case-sensitive prefix searching on strings
   </td>
  </tr>
  <tr>
   <td>ENDS WITH
   </td>
   <td>Case-sensitive suffix searching on strings
   </td>
  </tr>
  <tr>
   <td>CONTAINS
   </td>
   <td>Case-sensitive inclusion searching on strings
   </td>
  </tr>
  <tr>
   <td>=~
   </td>
   <td>Regular expression string matching
   </td>
  </tr>
  <tr>
   <td>4
   </td>
   <td>-
   </td>
   <td>Unary Minus
   </td>
  </tr>
  <tr>
   <td rowspan="3" >5
   </td>
   <td>IN
   </td>
   <td>Checking if an element exists in a list
   </td>
  </tr>
  <tr>
   <td>IS NULL
   </td>
   <td>Checking a value is NULL
   </td>
  </tr>
  <tr>
   <td>IS NOT NULL
   </td>
   <td>Checking a value is not NULL
   </td>
  </tr>
  <tr>
   <td>6
   </td>
   <td>^
   </td>
   <td>Exponentiation
   </td>
  </tr>
  <tr>
   <td>7
   </td>
   <td>* / %
   </td>
   <td>Multiplication, division and remainder
   </td>
  </tr>
  <tr>
   <td>8
   </td>
   <td>+ -
   </td>
   <td>Addition and Subtraction
   </td>
  </tr>
  <tr>
   <td rowspan="3" >9
   </td>
   <td>= &lt;>
   </td>
   <td>For relational = and ≠ respectively
   </td>
  </tr>
  <tr>
   <td>&lt; &lt;=
   </td>
   <td>For relational &lt; and ≤ respectively
   </td>
  </tr>
  <tr>
   <td>> >=
   </td>
   <td>For relational > and ≥ respectively
   </td>
  </tr>
  <tr>
   <td>10
   </td>
   <td>NOT
   </td>
   <td>Logical NOT
   </td>
  </tr>
  <tr>
   <td>11
   </td>
   <td>AND
   </td>
   <td>Logical AND
   </td>
  </tr>
  <tr>
   <td>12
   </td>
   <td>OR
   </td>
   <td>Logical OR
   </td>
  </tr>
</table>


# File: age-website-master/docs/intro/setup.md
# Setup

## Getting Apache AGE

### Releases

The releases and release notes can be found at [Apache AGE Releases](https://github.com/apache/age/releases).

### Source Code

The source code can be found at [Apache AGE GitHub Repository](https://github.com/apache/age).

## Installing From Source Code

### Pre-Installation

Before building Apache AGE from source, ensure that the following essential libraries are installed based on your operating system:

#### CentOS

```bash
yum install gcc glibc glib-common readline readline-devel zlib zlib-devel flex bison
```

#### Fedora

```bash
dnf install gcc glibc bison flex readline readline-devel zlib zlib-devel
```

#### Ubuntu

```bash
sudo apt-get install build-essential libreadline-dev zlib1g-dev flex bison
```

### Install PostgreSQL

You will need to install a PostgreSQL version compatible with Apache AGE. Apache AGE supports PostgreSQL versions 11, 12, 13, 14, and 15.

#### Install From Source Code

You can download the PostgreSQL source code from [PostgreSQL Downloads](https://www.postgresql.org/download/) and install your own instance of PostgreSQL. Refer to the [official PostgreSQL installation guide](https://www.postgresql.org/docs/15/installation.html) for instructions on installing from source code.

#### Install From a Package Manager

You can use a package manager provided by your operating system to download and install PostgreSQL.

##### Ubuntu

```bash
sudo apt install postgresql-15 postgresql-server-dev-all
```

Replace `15` with the desired PostgreSQL version if different.

### Installation

Clone the [Apache AGE GitHub repository](https://github.com/apache/age) or [download an official release](https://github.com/apache/age/releases).

Navigate to the source code directory of Apache AGE and run the following command to build and install the extension:

```bash
make install
```

If the path to your PostgreSQL installation is not in the PATH variable, specify the path to `pg_config` using the `PG_CONFIG` argument:

```bash
make PG_CONFIG=/path/to/postgres/bin/pg_config install
```

## Installing via Docker Image

### Get the Docker Image

```bash
docker pull apache/age
```

### Run Apache AGE Container

```bash
docker run \
    --name myPostgresDb  \
    -p 5455:5432 \
    -e POSTGRES_USER=postgresUser \
    -e POSTGRES_PASSWORD=postgresPW \
    -e POSTGRES_DB=postgresDB \
    -d \
    apache/age
```

| Docker Variables | Description                                        |
| ---------------- | -------------------------------------------------- |
| `--name `        | Assign a name to the container                     |
| `-p`             | Publish container's port(s) to the host            |
| `-e`             | Set environment variables                          |
| `-d`             | Run container in background and print container ID |

## Post-Installation Setup

### Per Session Instructions

For every connection to Apache AGE, load the AGE extension:

```sql
LOAD 'age';
```

Add `ag_catalog` to the `search_path` to simplify queries:

```sql
SET search_path = ag_catalog, "$user", public;
```

### Allow Non-Superusers to Use Apache AGE

To allow non-superusers to use Apache AGE:

1. Create a symlink to allow non-superusers to load the Apache AGE library:
   
   ```bash
   sudo ln -s /usr/lib/postgresql/15/lib/age.so /usr/lib/postgresql/15/lib/plugins/age.so
   ```

   Replace `/usr/lib/postgresql/15/lib/` with the appropriate path to the PostgreSQL library directory.

2. Grant `USAGE` privileges on the `ag_catalog` schema to the desired user (e.g., `db_user`):

   ```sql
   GRANT USAGE ON SCHEMA ag_catalog TO db_user;
   ```


# File: age-website-master/docs/intro/types.md
# Data Types - An Introduction to agtype

AGE uses a custom data type called agtype, which is the only data type returned by AGE. Agtype is a superset of Json and a custom implementation of JsonB.


## Simple Data Types


### Null

In Cypher, `null` is used to represent missing or undefined values. Conceptually, `null` means 'a missing unknown value' and it is treated somewhat differently from other values. For example getting a property from a vertex that does not have said property produces `null`. Most expressions that take `null` as input will produce `null`. This includes boolean expressions that are used as predicates in the `WHERE` clause. In this case, anything that is not true is interpreted as being false. `null` is not equal to `null`. Not knowing two values does not imply that they are the same value. So the expression `null = null` yields `null` and not true.

Input/Output Format

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN NULL
$$) AS (null_result agtype);
```


A null will appear as an empty space.

Result:


<table>
  <tr>
   <td>null_result
   </td>
  </tr>
  <tr>
   <td>
    <br>
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



#### Agtype NULL vs Postgres NULL

The concept of `NULL` in Agtype and Postgres is the same as it is in Cypher.

### Integer

The integer type stores whole numbers, i.e. numbers without fractional components. Integer data type is a 64-bit field that stores values from -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807. Attempts to store values outside this range will result in an error.

The type integer is the common choice, as it offers the best balance between range, storage size, and performance. The `smallint` type is generally used only if disk space is at a premium. The `bigint` type is designed to be used when the range of the integer type is insufficient.

Input/Output Format

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN 1
$$) AS (int_result agtype);
```


Result:


<table>
  <tr>
   <td>int_result
   </td>
  </tr>
  <tr>
   <td>1
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



### Float

The data type `float` is an inexact, variable-precision numeric type, conforming to the IEEE-754 Standard. 

Inexact means that some values cannot be converted exactly to the internal format and are stored as approximations, so that storing and retrieving a value might show slight discrepancies. Managing these errors and how they propagate through calculations is the subject of an entire branch of mathematics and computer science and will not be discussed here, except for the following points:


* If you require exact storage and calculations (such as for monetary amounts), use the numeric type instead.

* If you want to do complicated calculations with these types for anything important, especially if you rely on certain behavior in boundary cases (infinity, underflow), you should evaluate the implementation carefully.

* Comparing two floating-point values for equality might not always work as expected.


Values that are too large or too small will cause an error. Rounding might take place if the precision of an input number is too high. Numbers too close to zero that are not representable as distinct from zero will cause an underflow error.

In addition to ordinary numeric values, the floating-point types have several special values:
* Infinity
* -Infinity
* NaN

These represent the IEEE 754 special values “infinity”, “negative infinity”, and “not-a-number”, respectively. When writing these values as constants in a Cypher command, you must put quotes around them and typecast them, for example 
```
SET x.float_value = '-Infinity'::float
```
On input, these strings are recognized in a case-insensitive manner.

_Note
IEEE754 specifies that NaN should not compare equal to any other floating-point value (including NaN). However, in order to allow floats to be sorted correctly, AGE evaluates 'NaN'::float = 'NaN'::float to true. See the section Comparability and Equality for more details._

<!-- changed 'NaN':float to Nan::float -->

Input/Output Format:

To use a float, denote a decimal value.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN 1.0
$$) AS (float_result agtype);
```


Result:


<table>
  <tr>
   <td>float_result
   </td>
  </tr>
  <tr>
   <td>1.0
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



### Numeric 

The type `numeric` can store numbers with a very large number of digits. It is especially recommended for storing monetary amounts and other quantities where exactness is required. Calculations with numeric values yield exact results where possible, e.g., addition, subtraction, multiplication. However, calculations on numeric values are very slow compared to the integer types, or to the floating-point type.

We use the following terms below: The _precision_ of a numeric is the total count of significant digits in the whole number, that is, the number of digits to both sides of the decimal point. The _scale_ of a numeric is the count of decimal digits in the fractional part, to the right of the decimal point. So the number 23.5141 has a precision of 6 and a scale of 4. Integers can be considered to have a scale of zero.

Without any precision or scale creates a column in which numeric values of any precision and scale can be stored, up to the implementation limit on precision. <!-- fix above sentence --> A column of this kind will not coerce input values to any particular scale, whereas numeric columns with a declared scale will coerce input values to that scale. (The SQL standard requires a default scale of 0, i.e., coercion to integer precision. We find this a bit useless. If you're concerned about portability, always specify the precision and scale explicitly.)


```
_Note
The maximum allowed precision when explicitly specified in the type declaration is 1000; NUMERIC without a specified precision is subject to the limits described in Table 8.2._

```

If the scale of a value to be stored is greater than the declared scale of the column, the system will round the value to the specified number of fractional digits. Then, if the number of digits to the left of the decimal point exceeds the declared precision minus the declared scale, an error is raised.

Numeric values are physically stored without any extra leading or trailing zeroes. Thus, the declared precision and scale of a column are maximums, not fixed allocations. (In this sense the numeric type is more akin to `varchar(n)` than to `char(n)`.) The actual storage requirement is two bytes for each group of four decimal digits, plus three to eight bytes overhead.

In addition to ordinary numeric values, the numeric type allows the special value NaN, meaning “not-a-number”. Any operation on NaN yields another NaN. When writing this value as a constant in an SQL command, you must put quotes around it, for example UPDATE table SET x = 'NaN'. 



```
_Note
In most implementations of the "not-a-number" concept, NaN is considered not equal to any other numeric value (including NaN). However, in order to allow floats to be sorted correctly, AGE evaluates 'NaN'::numeric = 'NaN':numeric to true. See the section Comparability and Equality for more details._

```

When rounding values, the numeric type rounds ties away from zero, while (on most machines) the real and double precision types round ties to the nearest even number. For example:

Input/Output Format:

When creating a numeric data type, the `::numeric` data annotation is required.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN 1.0::numeric
$$) AS (numeric_result agtype);
```


Result:


<table>
  <tr>
   <td>numeric_result
   </td>
  </tr>
  <tr>
   <td>1.0::numeric
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



#### Bool 

AGE provides the standard Cypher type boolean. The boolean type can have several states: “true”, “false”, and a third state, “unknown”, which is represented by the Agtype null value.

Boolean constants can be represented in Cypher queries by the keywords `TRUE`, `FALSE`, and `NULL`.

Input/Output Format

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN TRUE
$$) AS (boolean_result agtype);
```


Unlike Postgres, AGE’s boolean outputs as the full word, ie. true and false as opposed to t and f.

Result:


<table>
  <tr>
   <td>boolean_result
   </td>
  </tr>
  <tr>
   <td>true
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



### String

Agtype strings String literals can contain the following escape sequences:


<table>
  <tr>
   <td>Escape Sequence
   </td>
   <td>Character
   </td>
  </tr>
  <tr>
   <td>\t
   </td>
   <td>Tab
   </td>
  </tr>
  <tr>
   <td>\b
   </td>
   <td>Backspace
   </td>
  </tr>
  <tr>
   <td>\n
   </td>
   <td>Newline
   </td>
  </tr>
  <tr>
   <td>\r
   </td>
   <td>Carriage Return
   </td>
  </tr>
  <tr>
   <td>\f
   </td>
   <td>Form Feed
   </td>
  </tr>
  <tr>
   <td>\’
   </td>
   <td>Single Quote
   </td>
  </tr>
  <tr>
   <td>\”
   </td>
   <td>Double Quote
   </td>
  </tr>
  <tr>
   <td>\\
   </td>
   <td>Backslash
   </td>
  </tr>
  <tr>
   <td>\uXXXX
   </td>
   <td>Unicode UTF-16 code point (4 hex digits must follow the \u)
   </td>
  </tr>
</table>


Input/Output Format

Use single (‘) quotes to identify a string. The output will use double (“) quotes.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    RETURN 'This is a string'
$$) AS (string_result agtype);
```


Result:


<table>
  <tr>
   <td>string_result
   </td>
  </tr>
  <tr>
   <td>“This is a string”
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



## Composite Data Types


### List

All examples will use the [`WITH`](../clauses/with.md) clause and [`RETURN`](../clauses/return.md) clause.


#### Lists in general

A literal list is created by using brackets and separating the elements in the list with commas.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    WITH [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] as lst
    RETURN lst
$$) AS (lst agtype);
```


Result:


<table>
  <tr>
   <td>lst
   </td>
  </tr>
  <tr>
   <td>[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



#### NULL in a List

A list can hold the value `null`, unlike when a `null` is an independent value, it will appear as the word ‘null’ in a list

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    WITH [null] as lst
    RETURN lst
$$) AS (lst agtype);
```


Result:


<table>
  <tr>
   <td>lst
   </td>
  </tr>
  <tr>
   <td>[null]
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



#### Access Individual Elements

To access individual elements in the list, we use the square brackets again. This will extract from the start index and up to but not including the end index.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    WITH [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] as lst
    RETURN lst[3]
$$) AS (element agtype);
```


Result:


<table>
  <tr>
   <td>element
   </td>
  </tr>
  <tr>
   <td>3
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



#### Map Elements in Lists

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
   WITH [0, {key: 'key_value'}, 2, 3, 4, 5, 6, 7, 8, 9, 10] as lst
    RETURN lst
$$) AS (map_value agtype);
```


Result:


<table>
  <tr>
   <td>map_value
   </td>
  </tr>
  <tr>
   <td>[0, {"key": "key_value"}, 2, 3, 4, 5, 6, 7, 8, 9, 10]
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



#### Accessing Map Elements in Lists

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
   WITH [0, {key: 'key_value'}, 2, 3, 4, 5, 6, 7, 8, 9, 10] as lst
    RETURN lst[1].key
$$) AS (map_value agtype);
```


Result:


<table>
  <tr>
   <td>map_value
   </td>
  </tr>
  <tr>
   <td>“key_value”
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



#### Negative Index Access

You can also use negative numbers, to start from the end of the list instead.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    WITH [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] as lst
    RETURN lst[-3]
$$) AS (element agtype);
```


Result:


<table>
  <tr>
   <td>element
   </td>
  </tr>
  <tr>
   <td>8
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



#### Index Ranges

Finally, you can use ranges inside the brackets to return ranges of the list.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    WITH [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] as lst
    RETURN lst[0..3]
$$) AS (element agtype);
```


Result:


<table>
  <tr>
   <td>element
   </td>
  </tr>
  <tr>
   <td>[0, 1, 2]
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



#### Negative Index Ranges

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    WITH [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] as lst
    RETURN lst[0..-5]
$$) AS (lst agtype);
```


Result:


<table>
  <tr>
   <td>lst
   </td>
  </tr>
  <tr>
   <td>[0, 1, 2, 3, 4, 5]
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



#### Positive Slices

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    WITH [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] as lst
    RETURN lst[..4]
$$) AS (lst agtype);
```


Result:


<table>
  <tr>
   <td>lst
   </td>
  </tr>
  <tr>
   <td>[0, 1, 2, 3]
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



#### Negative Slices

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    WITH [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] as lst
    RETURN lst[-5..]
$$) AS (lst agtype);
```


Result:


<table>
  <tr>
   <td>lst
   </td>
  </tr>
  <tr>
   <td>[6, 7, 8, 9, 10]
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>


Out-of-bound slices are simply truncated, but out-of-bound single elements return null.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    WITH [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] as lst
    RETURN lst[15]
$$) AS (element agtype);
```


Result:


<table>
  <tr>
   <td>element
   </td>
  </tr>
  <tr>
   <td>
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>


Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    WITH [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] as lst
    RETURN lst[5..15]
$$) AS (element agtype);
```


Result:


<table>
  <tr>
   <td>element
   </td>
  </tr>
  <tr>
   <td>[5, 6, 7, 8, 9, 10]
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



### Map

Maps can be constructed using Cypher.


#### Literal Maps with Simple Data Types

You can construct a simple map with simple agtypes

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    WITH {int_key: 1, float_key: 1.0, numeric_key: 1::numeric, bool_key: true, string_key: 'Value'} as m
    RETURN m
$$) AS (m agtype);
```


Result:


<table>
  <tr>
   <td>m
   </td>
  </tr>
  <tr>
   <td>{"int_key": 1, "bool_key": true, "float_key": 1.0, "string_key": "Value", "numeric_key": 1::numeric}
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



#### Literal Maps with Composite Data Types

A map can also contain Composite Data Types, i.e. lists and other maps.

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    WITH {listKey: [{inner: 'Map1'}, {inner: 'Map2'}], mapKey: {i: 0}} as m
    RETURN m
$$) AS (m agtype);
```


Result:


<table>
  <tr>
   <td>m
   </td>
  </tr>
  <tr>
   <td>{"mapKey": {"i": 0}, "listKey": [{"inner": "Map1"}, {"inner": "Map2"}]}
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



#### Property Access of a map

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    WITH {int_key: 1, float_key: 1.0, numeric_key: 1::numeric, bool_key: true, string_key: 'Value'} as m
    RETURN m.int_key
$$) AS (int_key agtype);
```


Result:


<table>
  <tr>
   <td>int_key
   </td>
  </tr>
  <tr>
   <td>1
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



#### Accessing List Elements in Maps

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
    WITH {listKey: [{inner: 'Map1'}, {inner: 'Map2'}], mapKey: {i: 0}} as m
    RETURN m.listKey[0]
$$) AS (m agtype);
```


Result:


<table>
  <tr>
   <td>m
   </td>
  </tr>
  <tr>
   <td>{"inner": "Map1"}
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



## Simple Entities

An entity has a unique, comparable identity which defines whether or not two entities are equal.

An entity is assigned a set of properties, each of which are uniquely identified in the set by the irrespective property keys.


### GraphId

Simple entities are assigned a unique graphid. A graphid is a unique composition of the entity's label id and a unique sequence assigned to each label. Note that there will be overlap in ids when comparing entities from different graphs.


### Labels

A label is an identifier that classifies vertices and edges into certain categories.



* Edges are required to have a label, but vertices do not. 
* The names of labels between vertices and edges cannot overlap. 

See [CREATE](../clauses/create.md) clause for information about how to make entities with labels.


### Properties

Both vertices and edges may have properties. Properties are attribute values, and each attribute name should be defined only as a string type. 


## Vertex



* A vertex is the basic entity of the graph, with the unique attribute of being able to exist in and ofitself.
* A vertex may be assigned a label.
* A vertex  may have zero or more outgoing edges.
* A vertex may have zero or more incoming edges.

Data Format:


<table>
  <tr>
   <td><strong>Attribute Name</strong>
   </td>
   <td><strong>Description</strong>
   </td>
  </tr>
  <tr>
   <td>Id
   </td>
   <td>graphid for this vertex
   </td>
  </tr>
  <tr>
   <td>label
   </td>
   <td>Name of the label this vertex has
   </td>
  </tr>
  <tr>
   <td>properties
   </td>
   <td>Properties associated with this vertex
   </td>
  </tr>
</table>



```
{id:1; label: 'label_name'; properties: {prop1: value1, prop2: value2}}::vertex
```



### Type Casting a Map to a Vertex

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
	WITH {id: 0, label: "label_name", properties: {i: 0}}::vertex as v
	RETURN v
$$) AS (v agtype);
```


Result:
<table>
  <tr>
   <td>v
   </td>
  </tr>
  <tr>
   <td>{"id": 0, "label": "label_name", "properties": {"i": 0}}::vertex
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>





## Edge

An edge is an entity that encodes a directed connection between exactly two nodes, the source node and the target node. An outgoing edge is a directed relationship from the point of view of its source node. An incoming edge is a directed relationship from the point of view of its target node. An edge is assigned exactly one edge type.

Data Format


<table>
  <tr>
   <td><strong>Attribute Name</strong>
   </td>
   <td><strong>Description</strong>
   </td>
  </tr>
  <tr>
   <td>id
   </td>
   <td>graphid for this edge
   </td>
  </tr>
  <tr>
   <td>startid
   </td>
   <td>graphid for the source node
   </td>
  </tr>
  <tr>
   <td>endid
   </td>
   <td>graphid for the target node
   </td>
  </tr>
  <tr>
   <td>label
   </td>
   <td>Name of the label this edge has
   </td>
  </tr>
  <tr>
   <td>properties
   </td>
   <td>Properties associated with this edge 
   </td>
  </tr>
</table>


Output:


```
{id: 3; startid: 1; endid: 2; label: 'edge_label' properties{prop1: value1, prop2: value2}}::edge
```



### Type Casting a Map to an Edge

Query


```postgresql
SELECT *
FROM cypher('graph_name', $$
	WITH {id: 2, start_id: 0, end_id: 1, label: "label_name", properties: {i: 0}}::edge as e
	RETURN e
$$) AS (e agtype);
```


Result:


<table>
  <tr>
   <td>v
   </td>
  </tr>
  <tr>
   <td>{"id": 2, "label": "label_name", "end_id": 1, "start_id": 0, "properties": {"i": 0}}::edge
   </td>
  </tr>
  <tr>
   <td>(1 row)
   </td>
  </tr>
</table>



## Composite Entities


### Path

A path is a series of alternating vertices and edges. A path must start with a vertex, and have at least one edge.


#### Type Casting a List to a Path

Query

```postgresql
SELECT *
FROM cypher('graph_name', $$
	WITH [{id: 0, label: "label_name_1", properties: {i: 0}}::vertex,
            {id: 2, start_id: 0, end_id: 1, label: "edge_label", properties: {i: 0}}::edge,
           {id: 1, label: "label_name_2", properties: {}}::vertex
           ]::path as p
	RETURN p
$$) AS (p agtype);
```


The result is formatted to improve readability

Result:

<table>
   <tr>
      <td>p
      </td>
   </tr>
   <tr>
      <td>[{"id": 0, "label": "label_name_1", "properties": {"i": 0}}::vertex, {"id": 2, "label": "edge_label", "end_id": 1, "start_id": 0, "properties": {"i": 0}}::edge, <br> {"id": 1, "label": "label_name_2", "properties": {}}::vertex]::path
      </td>
   </tr>
   <tr>
      <td>(1 row)
      </td>
   </tr>
</table>




