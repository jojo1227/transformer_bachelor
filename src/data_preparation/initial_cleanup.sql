/* drop redundant tables */

drop table event_import;
drop table host;


/* drop duplicate entries from principal */
delete from principal
where ctid not in (
    select distinct on (uuid, type, hostid, userid, groupids, username_string) ctid
    from principal
    order by uuid, type, hostid, userid, groupids, username_string, ctid
);




/* drop import columns from tables */
alter table event drop column line;
alter table event drop column line_no;
alter table fileobject drop column line;
alter table fileobject drop column line_no;
alter table netflowobject drop column line;
alter table netflowobject drop column line_no;
alter table node_uuids drop column line_no;
alter table principal drop column line;
alter table principal drop column line_no;
alter table srcsinkobject drop column line;
alter table srcsinkobject drop column line_no;
alter table subject drop column line;
alter table subject drop column line_no;
alter table unnamedpipeobject drop column line;
alter table unnamedpipeobject drop column line_no;


/* drop empty columns */
alter table event drop column predicateobject2path, drop column predicateobjectpath, drop column size, drop column predicateobject2, drop column predicateobject, drop column subject, drop column name, drop column parameters, drop column hostid, drop column location, drop column programpoint, drop column properties_map_host;
alter table fileobject drop column filedescriptor, drop column localprincipal, drop column size, drop column peinfo, drop column hashes, drop column baseobject_hostid, drop column baseobject_permission, drop column baseobject_epoch;
alter table netflowobject drop column ipprotocol, drop column filedescriptor, drop column baseobject_hostid, drop column baseobject_permission, drop column baseobject_epoch;
alter table principal drop column type, drop column hostid, drop column groupids;
alter table srcsinkobject drop column type, drop column filedescriptor, drop column baseobject_hostid, drop column baseobject_permission, drop column baseobject_epoch;
alter table subject drop column hostid, drop column unitid, drop column iteration, drop column count, drop column cmdline, drop column privilegelevel, drop column importedlibraries, drop column exportedlibraries, drop column properties_map_host, drop column parentsubject, drop column type;
alter table unnamedpipeobject drop column sourcefiledescriptor, drop column sinkfiledescriptor, drop column baseobject_hostid, drop column baseobject_permission, drop column baseobject_epoch;